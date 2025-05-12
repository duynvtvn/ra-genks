import logging
import os
import json
import re
import torch
import numpy as np
import faiss
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DPRContextEncoder,
    DPRQuestionEncoder,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import math

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HyperlinkProcessor:
    """
    Lớp xử lý và quản lý hyperlink giữa đối thoại và tri thức

    Hyperlink là cơ chế quan trọng trong GenKS để nắm bắt mối liên kết giữa các lượt đối thoại
    với các đoạn tri thức, giúp mô hình hiểu được cấu trúc diễn ngôn và duy trì sự liên tục
    trong việc sử dụng tri thức qua các lượt đối thoại.
    """

    def __init__(self):
        """Khởi tạo bộ xử lý hyperlink"""
        # Lưu trữ mapping giữa turn_id và tri thức (knowledge_id, title)
        self.dialogue_knowledge_mapping = {}

    def add_hyperlink(self, turn_id, knowledge_id, title):
        """
        Thêm hyperlink giữa lượt đối thoại và đoạn tri thức

        Args:
            turn_id: ID của lượt đối thoại
            knowledge_id: ID của đoạn tri thức (ví dụ: 'k1')
            title: Tiêu đề của đoạn tri thức
        """
        self.dialogue_knowledge_mapping[turn_id] = (knowledge_id, title)

    def get_hyperlinks_for_context(self, dialogue_history):
        """
        Lấy lịch sử đối thoại đã được bổ sung hyperlink

        Args:
            dialogue_history: Danh sách các phát ngôn trong lịch sử đối thoại

        Returns:
            Danh sách các phát ngôn đã được bổ sung hyperlink
        """
        hyperlinked_history = []

        for i, utterance in enumerate(dialogue_history):
            if i in self.dialogue_knowledge_mapping:
                knowledge_id, title = self.dialogue_knowledge_mapping[i]
                # Thêm hyperlink vào đầu phát ngôn, theo định dạng [title]<knowledge_id>
                hyperlinked_utterance = f"[{title}]<{knowledge_id}> {utterance}"
                hyperlinked_history.append(hyperlinked_utterance)
            else:
                hyperlinked_history.append(utterance)

        return hyperlinked_history


class GenerativeKnowledgeSelector:
    """
    Bộ chọn tri thức dựa trên phương pháp tạo sinh (generative approach)

    Thay vì phân loại độc lập từng đoạn tri thức, lớp này sử dụng mô hình sequence-to-sequence
    để tạo sinh định danh của các đoạn tri thức liên quan nhất, giúp nắm bắt mối quan hệ
    giữa các đoạn tri thức và cải thiện khả năng hiểu ngữ cảnh.
    """

    def __init__(self, model, tokenizer, max_knowledge_candidates=20, device='cuda'):
        """
        Khởi tạo bộ chọn tri thức dựa trên phương pháp tạo sinh

        Args:
            model: Mô hình sequence-to-sequence
            tokenizer: Tokenizer tương ứng với mô hình
            max_knowledge_candidates: Số lượng ứng viên tri thức tối đa
            device: Thiết bị tính toán ('cuda' hoặc 'cpu')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_knowledge_candidates = max_knowledge_candidates
        self.device = device

    def prepare_input_for_generation(self, query, dialogue_history, knowledge_candidates, hyperlinked_history=None):
        """
        Chuẩn bị đầu vào cho mô hình tạo sinh để chọn tri thức

        Args:
            query: Câu truy vấn hiện tại
            dialogue_history: Lịch sử đối thoại
            knowledge_candidates: Danh sách các đoạn tri thức ứng viên
            hyperlinked_history: Lịch sử đối thoại đã bổ sung hyperlink

        Returns:
            Chuỗi đầu vào đã định dạng
        """
        # Xây dựng ngữ cảnh đối thoại
        context_parts = []

        # Sử dụng lịch sử đã bổ sung hyperlink nếu có
        history_to_use = hyperlinked_history if hyperlinked_history else dialogue_history

        if history_to_use:
            for i, utterance in enumerate(history_to_use):
                speaker = "User1: " if i % 2 == 0 else "User2: "
                context_parts.append(f"{speaker}{utterance}")

        # Thêm truy vấn hiện tại
        current_speaker = "User1: " if len(context_parts) % 2 == 0 else "User2: "
        context_parts.append(f"{current_speaker}{query}")

        context_text = "\n".join(context_parts)

        # Xây dựng phần tri thức
        knowledge_parts = ["Thông tin tham khảo:"]

        # Giới hạn số lượng ứng viên tri thức để tránh vượt quá độ dài tối đa
        candidates_to_use = knowledge_candidates[:self.max_knowledge_candidates]

        for i, (title, text, _) in enumerate(candidates_to_use):
            # Gán định danh cho từng đoạn tri thức
            knowledge_parts.append(f"<k{i + 1}> [{title}] {text}")

        knowledge_text = "\n".join(knowledge_parts)

        # Kết hợp ngữ cảnh và tri thức
        # Phần prompt đặc biệt "Chọn tri thức phù hợp nhất:" sẽ hướng dẫn mô hình sinh ra định danh
        input_text = f"{context_text}\n\n{knowledge_text}\n\nChọn tri thức phù hợp nhất:"

        return input_text

    def select_knowledge(self, query, knowledge_candidates, dialogue_history=None, hyperlinked_history=None, top_k=3):
        """
        Chọn top-k tri thức phù hợp nhất bằng cách tạo sinh định danh

        Args:
            query: Câu truy vấn hiện tại
            knowledge_candidates: Danh sách các đoạn tri thức ứng viên [(title, text, score)]
            dialogue_history: Lịch sử đối thoại
            hyperlinked_history: Lịch sử đối thoại đã bổ sung hyperlink
            top_k: Số lượng đoạn tri thức cần chọn

        Returns:
            Danh sách các đoạn tri thức được chọn [(title, text, score)]
        """
        if not knowledge_candidates:
            return []

        # Chuẩn bị đầu vào
        input_text = self.prepare_input_for_generation(
            query, dialogue_history, knowledge_candidates, hyperlinked_history
        )

        # Mã hóa đầu vào
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        # Sinh định danh tri thức
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=20,  # Sinh ngắn gọn chỉ các định danh
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        # Giải mã đầu ra để lấy các định danh
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        logger.info(f"Generated identifier text: {generated_text}")

        # Trích xuất các định danh được sinh ra (<k1>, <k2>, v.v.)
        selected_knowledge = []
        pattern = r'<k(\d+)>'
        matches = re.findall(pattern, generated_text)

        # Chuyển định danh thành chỉ số (id) và lấy tri thức tương ứng
        for match in matches:
            idx = int(match) - 1  # Chuyển từ 1-indexed sang 0-indexed
            if idx < len(knowledge_candidates):
                title, text, score = knowledge_candidates[idx]
                selected_knowledge.append((title, text, score))
                if len(selected_knowledge) >= top_k:
                    break

        # Nếu không tìm thấy đủ định danh, bổ sung bằng cách lấy top-k theo điểm số
        if len(selected_knowledge) < top_k:
            # Sắp xếp tri thức còn lại theo điểm số và bổ sung
            remaining_candidates = [
                c for c in knowledge_candidates
                if all(c[1] != k[1] for k in selected_knowledge)
            ]
            remaining_candidates.sort(key=lambda x: x[2], reverse=True)

            # Thêm vào danh sách đã chọn
            selected_knowledge.extend(
                remaining_candidates[:top_k - len(selected_knowledge)]
            )

        return selected_knowledge


class ImprovedMultiSpanGENKSData(Dataset):
    """
    Lớp xử lý dữ liệu cho mô hình GenKS cải tiến với hỗ trợ hyperlink và chọn nhiều đoạn tri thức
    """

    def __init__(self, data, tokenizer, context_len=256, knowledge_len=64, max_length=1024,
                 test=False, top_k_knowledge=3, add_hyperlink=True):
        """
        Khởi tạo bộ dữ liệu

        Args:
            data: Dữ liệu đầu vào
            tokenizer: Tokenizer để xử lý văn bản
            context_len: Độ dài tối đa của ngữ cảnh
            knowledge_len: Độ dài tối đa của mỗi đoạn tri thức
            max_length: Độ dài tối đa của chuỗi đầu vào
            test: Chế độ kiểm thử
            top_k_knowledge: Số lượng đoạn tri thức tối đa sử dụng
            add_hyperlink: Có thêm hyperlink hay không
        """
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.knowledge_len = knowledge_len
        self.max_length = max_length
        self.test = test
        self.top_k_knowledge = top_k_knowledge
        self.add_hyperlink = add_hyperlink

        # Hyperlink processor để quản lý và xây dựng hyperlink
        self.hyperlink_processor = HyperlinkProcessor()

        # Thêm đánh dấu đặc biệt cho tri thức
        special_tokens = ['<knowledge>', '</knowledge>']
        for i in range(1, top_k_knowledge + 1):
            special_tokens.extend([f'<k{i}>', f'</k{i}>'])

        self.tokenizer.add_tokens(special_tokens)

    def __getitem__(self, index):
        """
        Lấy mẫu dữ liệu tại vị trí index

        Args:
            index: Vị trí của mẫu dữ liệu

        Returns:
            Tuple (input_ids, labels)
        """
        example = self.data[index]

        # =============================
        # Xử lý ngữ cảnh đối thoại
        # =============================
        context_parts = []

        # Thêm thông tin chủ đề nếu có
        if 'chosen_topic' in example:
            context_parts.append(f"Chủ đề: {example['chosen_topic']}")

        # Xử lý lịch sử đối thoại
        dialogue_history = []
        hyperlinked_history = []

        # Mapping roles
        role = {'0_Wizard': 'User1', '1_Apprentice': 'User2', '0_Apprentice': 'User2',
                '1_Wizard': 'User1', 0: 'User1', 1: 'User2', 'user1': 'User1', 'user2': 'User2'}

        if 'context' in example:
            for i, turn in enumerate(example['context']):
                speaker = role.get(turn.get('speaker', ''), turn.get('speaker', ''))
                text = turn.get('text', '')
                dialogue_history.append(text)

                # Thêm hyperlink nếu cần và có tri thức được sử dụng
                if self.add_hyperlink and 'knowledge_used' in turn:
                    knowledge_id = f"k{i + 1}"
                    knowledge_title = turn.get('knowledge_title', 'unknown')
                    self.hyperlink_processor.add_hyperlink(i, knowledge_id, knowledge_title)

            # Lấy lịch sử với hyperlink
            if self.add_hyperlink:
                hyperlinked_history = self.hyperlink_processor.get_hyperlinks_for_context(dialogue_history)

        # Sử dụng lịch sử với hyperlink nếu có
        history_to_use = hyperlinked_history if hyperlinked_history else dialogue_history

        # Thêm lịch sử đối thoại vào ngữ cảnh
        for i, utterance in enumerate(history_to_use):
            speaker = role.get(i % 2, f"User{(i % 2) + 1}")
            context_parts.append(f"{speaker}: {utterance}")

        # Kết hợp thành ngữ cảnh hoàn chỉnh
        context_text = "\n".join(context_parts)

        # =============================
        # Xử lý tri thức
        # =============================
        # Thu thập các đoạn tri thức
        knowledge_items = []

        # Lấy tri thức từ ví dụ
        if 'knowledge' in example:
            for title, sentences in example['knowledge'].items():
                for sentence in sentences:
                    knowledge_items.append((title, sentence, 1.0))  # Mặc định điểm số 1.0

        # Ưu tiên tri thức được sử dụng nếu có
        checked_knowledge = []
        if 'title' in example and example.get('title') != 'no_passages_used' and 'checked_sentence' in example:
            checked_knowledge.append((example['title'], example['checked_sentence'], 1.0))
            # Loại bỏ tri thức đã kiểm tra khỏi danh sách để tránh trùng lặp
            knowledge_items = [k for k in knowledge_items if k[1] != example['checked_sentence']]

        # Kết hợp tri thức đã kiểm tra và các tri thức khác
        selected_knowledge = checked_knowledge.copy()

        # Nếu cần thêm tri thức để đủ top_k
        if len(selected_knowledge) < self.top_k_knowledge and knowledge_items:
            # Sắp xếp ngẫu nhiên trong quá trình huấn luyện để tăng tính đa dạng
            if not self.test:
                np.random.shuffle(knowledge_items)

            # Thêm tri thức vào danh sách đã chọn
            selected_knowledge.extend(
                knowledge_items[:self.top_k_knowledge - len(selected_knowledge)]
            )

        # =============================
        # Xây dựng chuỗi đầu vào
        # =============================
        input_sequence = context_text

        # Thêm tri thức đã chọn
        if selected_knowledge:
            knowledge_parts = ["<knowledge>Thông tin tham khảo:"]

            for i, (title, sentence, _) in enumerate(selected_knowledge):
                if i < self.top_k_knowledge:
                    # Sử dụng thẻ đánh dấu tri thức đặc biệt
                    knowledge_parts.append(f"<k{i + 1}>[{title}] {sentence}</k{i + 1}>")

            knowledge_parts.append("</knowledge>")
            knowledge_text = "\n".join(knowledge_parts)

            # Kết hợp ngữ cảnh và tri thức
            input_sequence = f"{input_sequence}\n\n{knowledge_text}\n\nPhản hồi:"
        else:
            input_sequence = f"{input_sequence}\n\nPhản hồi:"

        # =============================
        # Xây dựng đầu ra (nhãn)
        # =============================
        if 'response' in example:
            target = example['response']
        elif 'labels' in example and example['labels']:
            target = example['labels'][0]
        else:
            target = ""

        # Mã hóa đầu vào và đầu ra
        input_ids = self.tokenizer.encode(
            input_sequence,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )

        labels = self.tokenizer.encode(
            target,
            truncation=True,
            max_length=self.context_len,
            add_special_tokens=True
        )

        return torch.tensor(input_ids), torch.tensor(labels)

    def __len__(self):
        """Trả về số lượng mẫu dữ liệu"""
        return len(self.data)

    def collate_fn(self, data):
        """
        Hàm gộp batch cho DataLoader

        Args:
            data: Danh sách các mẫu

        Returns:
            Dict batch đã gộp
        """
        from torch.nn.utils.rnn import pad_sequence
        padding_value = self.tokenizer.pad_token_id
        input_ids, labels = zip(*data)

        # Padding các chuỗi trong batch
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(padding_value),
            'labels': labels,
        }


class EnhancedGenKSWithRAG:
    """
    Mô hình GenKS cải tiến kết hợp với RAG, hỗ trợ:
    1. Phương pháp tạo sinh để chọn tri thức (theo GenKS)
    2. Cơ chế hyperlink để theo dõi mối liên hệ giữa đối thoại và tri thức
    3. Quy trình RAG đa giai đoạn để cải thiện chất lượng tri thức
    """

    def __init__(
            self,
            model_name='facebook/bart-base',
            retriever_model_name='facebook/dpr-ctx_encoder-single-nq-base',
            query_encoder_name='facebook/dpr-question_encoder-single-nq-base',
            ranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
            embed_dim=768,
            top_k_retrieval=100,
            top_k_rerank=20,
            top_k_knowledge=3,
            retrieval_method='bm25',
            use_generative_selection=True,
            device='cuda',
            cache_dir=None
    ):
        """
        Khởi tạo mô hình GenKS cải tiến

        Args:
            model_name: Tên mô hình ngôn ngữ chính
            retriever_model_name: Tên mô hình bộ truy xuất
            query_encoder_name: Tên mô hình mã hóa truy vấn
            ranker_model_name: Tên mô hình xếp hạng
            embed_dim: Chiều của embedding
            top_k_retrieval: Số lượng tài liệu truy xuất
            top_k_rerank: Số lượng tài liệu sau khi xếp hạng lại
            top_k_knowledge: Số lượng đoạn tri thức được chọn
            retrieval_method: Phương pháp truy xuất ('bm25' hoặc 'dpr')
            use_generative_selection: Có sử dụng phương pháp tạo sinh để chọn tri thức hay không
            device: Thiết bị tính toán
            cache_dir: Thư mục cache
        """
        self.device = device
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.top_k_knowledge = top_k_knowledge
        self.retrieval_method = retrieval_method.lower()
        self.use_generative_selection = use_generative_selection

        # Thiết lập tokenizer và mô hình chính
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Thêm tokens đặc biệt cho tri thức
        special_tokens = ['<knowledge>', '</knowledge>']
        for i in range(1, top_k_knowledge + 1):
            special_tokens.extend([f'<k{i}>', f'</k{i}>'])

        self.tokenizer.add_tokens(special_tokens)

        # Tải mô hình chính
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Thiết lập bộ truy xuất và bộ xếp hạng
        self._initialize_retrievers(retriever_model_name, query_encoder_name, embed_dim)
        self._initialize_ranker(ranker_model_name)

        # Thiết lập bộ chọn tri thức dựa trên phương pháp tạo sinh
        self.generative_selector = GenerativeKnowledgeSelector(
            model=self.model,
            tokenizer=self.tokenizer,
            max_knowledge_candidates=20,
            device=device
        )

        # Thiết lập bộ xử lý hyperlink
        self.hyperlink_processor = HyperlinkProcessor()

        # Bộ nhớ cache cho các tài liệu đã truy xuất
        self.doc_cache = {}
        self.corpus = None
        self.id_to_doc = None

    def _initialize_retrievers(self, retriever_model_name, query_encoder_name, embed_dim):
        """
        Khởi tạo các bộ truy xuất (DPR, BM25, SBERT, TF-IDF)

        Args:
            retriever_model_name: Tên mô hình bộ truy xuất
            query_encoder_name: Tên mô hình mã hóa truy vấn
            embed_dim: Chiều của embedding
        """
        # Luôn khởi tạo BM25 vì nó nhẹ và hiệu quả
        try:
            from rank_bm25 import BM25Okapi
            self.bm25_class = BM25Okapi
            self.bm25 = None  # Sẽ được khởi tạo trong build_corpus_index
            self.tokenized_corpus = None
            logger.info("Đã khởi tạo BM25 thành công")
        except ImportError:
            logger.warning("Không thể nhập rank_bm25. Cài đặt bằng cách 'pip install rank-bm25'")
            self.bm25_class = None

        # Nếu sử dụng DPR, khởi tạo mô hình DPR
        if self.retrieval_method == 'dpr':
            try:
                self.ctx_encoder = DPRContextEncoder.from_pretrained(retriever_model_name).to(self.device)
                self.query_encoder = DPRQuestionEncoder.from_pretrained(query_encoder_name).to(self.device)
                self.dpr_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
                self.query_tokenizer = AutoTokenizer.from_pretrained(query_encoder_name)
                logger.info("Đã khởi tạo bộ truy xuất DPR thành công")
            except Exception as e:
                logger.warning(f"Không thể khởi tạo DPR: {str(e)}, sẽ sử dụng SentenceTransformer thay thế")
                try:
                    self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(self.device)
                    self.index = faiss.IndexFlatIP(embed_dim)
                    logger.info("Đã khởi tạo SentenceTransformer thành công")
                except Exception as e:
                    logger.warning(f"Không thể khởi tạo SentenceTransformer: {str(e)}, sẽ sử dụng TF-IDF")
                    self.retrieval_method = 'tfidf'

        # Luôn khởi tạo TF-IDF làm phương pháp dự phòng
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    def _initialize_ranker(self, ranker_model_name):
        """
        Khởi tạo bộ xếp hạng lại (re-ranker)

        Args:
            ranker_model_name: Tên mô hình cross-encoder để xếp hạng
        """
        try:
            from sentence_transformers import CrossEncoder
            self.ranker = CrossEncoder(ranker_model_name)
            logger.info(f"Đã khởi tạo CrossEncoder thành công: {ranker_model_name}")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo CrossEncoder: {str(e)}, sẽ sử dụng phương pháp xếp hạng đơn giản")
            self.ranker = None

    def build_corpus_index(self, corpus, cache_path=None):
        """
        Xây dựng chỉ mục (index) cho corpus để truy xuất nhanh

        Args:
            corpus: Danh sách tài liệu [(id, text, title)]
            cache_path: Đường dẫn để lưu/tải embeddings từ cache
        """
        self.corpus = corpus
        self.id_to_doc = {item[0]: (item[1], item[2]) for item in corpus}
        texts = [doc[1] for doc in corpus]

        # Xây dựng chỉ mục BM25
        if hasattr(self, 'bm25_class') and self.bm25_class is not None:
            logger.info("Đang xây dựng chỉ mục BM25...")
            # Tokenize corpus cho BM25
            self.tokenized_corpus = [text.lower().split() for text in texts]
            self.bm25 = self.bm25_class(self.tokenized_corpus)
            logger.info("Đã xây dựng chỉ mục BM25 thành công")

        # Nếu sử dụng DPR, xây dựng chỉ mục DPR
        if self.retrieval_method == 'dpr':
            # Tải embeddings từ cache nếu có
            if cache_path and os.path.exists(cache_path):
                logger.info(f"Đang tải embeddings từ cache: {cache_path}")
                self.corpus_embeddings = torch.load(cache_path)
                # Xây dựng chỉ mục FAISS
                self.index = faiss.IndexFlatIP(self.corpus_embeddings.size(1))
                self.index.add(self.corpus_embeddings.cpu().numpy())
                logger.info("Đã tải và xây dựng chỉ mục từ cache thành công")
                return

            # Xây dựng embeddings mới nếu không có cache
            if hasattr(self, 'ctx_encoder') and hasattr(self, 'dpr_tokenizer'):
                logger.info("Đang tạo embeddings cho corpus với DPR...")
                self.corpus_embeddings = []

                batch_size = 32
                for i in tqdm(range(0, len(texts), batch_size)):
                    batch_texts = texts[i:i + batch_size]
                    with torch.no_grad():
                        inputs = self.dpr_tokenizer(
                            batch_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512
                        ).to(self.device)
                        embeddings = self.ctx_encoder(**inputs).pooler_output
                    self.corpus_embeddings.append(embeddings.cpu())

                self.corpus_embeddings = torch.cat(self.corpus_embeddings, dim=0)

                # Lưu embeddings vào cache nếu có đường dẫn
                if cache_path:
                    logger.info(f"Đang lưu embeddings vào cache: {cache_path}")
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(self.corpus_embeddings, cache_path)

                # Xây dựng chỉ mục FAISS
                self.index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
                self.index.add(self.corpus_embeddings.cpu().numpy())

            elif hasattr(self, 'encoder'):
                # Sử dụng SentenceTransformer
                logger.info("Đang tạo embeddings cho corpus với SentenceTransformer...")
                self.corpus_embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_tensor=True)

                # Xây dựng chỉ mục FAISS
                self.index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
                self.index.add(self.corpus_embeddings.cpu().numpy())

        # Xây dựng ma trận TF-IDF dự phòng
        logger.info("Đang tạo ma trận TF-IDF...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

    def retrieve_documents(self, query, dialogue_history=None, top_k=None):
        """
        Giai đoạn 1: Truy xuất tài liệu sơ bộ

        Args:
            query: Câu truy vấn
            dialogue_history: Lịch sử đối thoại
            top_k: Số lượng tài liệu cần truy xuất

        Returns:
            Danh sách tài liệu truy xuất [(id, text, title, score)]
        """
        if top_k is None:
            top_k = self.top_k_retrieval

        # Kết hợp truy vấn với lịch sử đối thoại nếu có
        if dialogue_history:
            last_n_turns = dialogue_history[-3:]  # Lấy 3 lượt cuối
            combined_query = " ".join(last_n_turns) + " " + query
        else:
            combined_query = query

        # Kiểm tra bộ nhớ cache
        cache_key = combined_query[:100]  # Sử dụng phần đầu của truy vấn làm khóa
        if cache_key in self.doc_cache:
            logger.info(f"Sử dụng kết quả từ bộ nhớ cache cho truy vấn: {cache_key}...")
            return self.doc_cache[cache_key]

        # Sử dụng BM25 nếu được chọn hoặc phương pháp khác không khả dụng
        if (self.retrieval_method == 'bm25' or self.retrieval_method not in ['dpr', 'tfidf']) and hasattr(self,
                                                                                                          'bm25') and self.bm25 is not None:
            logger.info("Đang truy xuất với BM25...")
            # Tokenize truy vấn tương tự corpus
            tokenized_query = combined_query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            indices = np.argsort(-scores)[:top_k]

            # Chuyển indices thành tài liệu
            retrieved_docs = []
            for idx in indices:
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(scores[idx])))

        # Sử dụng DPR nếu được chọn và khả dụng
        elif self.retrieval_method == 'dpr' and hasattr(self, 'query_encoder'):
            logger.info("Đang truy xuất với DPR...")
            with torch.no_grad():
                inputs = self.query_tokenizer(combined_query, return_tensors="pt").to(self.device)
                query_embedding = self.query_encoder(**inputs).pooler_output.cpu().numpy()

            # Tìm kiếm với FAISS
            scores, indices = self.index.search(query_embedding, top_k)

            # Chuyển indices thành tài liệu
            retrieved_docs = []
            for idx, score in zip(indices.flatten(), scores.flatten()):
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(score)))

        # Sử dụng SentenceTransformer nếu DPR không khả dụng
        elif hasattr(self, 'encoder'):
            logger.info("Đang truy xuất với SentenceTransformer...")
            query_embedding = self.encoder.encode(combined_query, convert_to_tensor=True)

            # Tìm kiếm với FAISS
            scores, indices = self.index.search(query_embedding.cpu().numpy().reshape(1, -1), top_k)

            # Chuyển indices thành tài liệu
            retrieved_docs = []
            for idx, score in zip(indices.flatten(), scores.flatten()):
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(score)))

        # Fallback sang TF-IDF nếu các phương pháp khác không khả dụng
        else:
            logger.info("Đang truy xuất với TF-IDF...")
            query_vec = self.tfidf_vectorizer.transform([combined_query])
            scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
            indices = np.argsort(-scores)[:top_k]

            # Chuyển indices thành tài liệu
            retrieved_docs = []
            for idx in indices:
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(scores[idx])))

        # Lưu vào cache
        self.doc_cache[cache_key] = retrieved_docs

        return retrieved_docs

    def rerank_documents(self, query, docs, dialogue_history=None, top_k=None):
        """
        Giai đoạn 2: Xếp hạng lại tài liệu

        Args:
            query: Câu truy vấn
            docs: Danh sách tài liệu đã truy xuất
            dialogue_history: Lịch sử đối thoại
            top_k: Số lượng tài liệu cần trả về

        Returns:
            Danh sách tài liệu đã xếp hạng lại [(id, text, title, score)]
        """
        if top_k is None:
            top_k = self.top_k_rerank

        # Kết hợp truy vấn với lịch sử đối thoại nếu có
        if dialogue_history:
            last_n_turns = dialogue_history[-3:]  # Lấy 3 lượt cuối
            combined_query = " ".join(last_n_turns) + " " + query
        else:
            combined_query = query

        # Sử dụng cross-encoder nếu có
        if self.ranker:
            logger.info("Đang xếp hạng lại tài liệu với CrossEncoder...")
            pairs = [(combined_query, doc[1]) for doc in docs]
            scores = self.ranker.predict(pairs)

            # Xếp hạng lại dựa trên điểm mới
            reranked_docs = [
                (docs[i][0], docs[i][1], docs[i][2], float(scores[i]))
                for i in range(len(docs))
            ]
            reranked_docs = sorted(reranked_docs, key=lambda x: x[3], reverse=True)[:top_k]
        else:
            # Nếu không có ranker, giữ nguyên thứ tự từ bước truy xuất
            logger.info("Không có CrossEncoder, sử dụng điểm truy xuất gốc để xếp hạng...")
            reranked_docs = sorted(docs, key=lambda x: x[3], reverse=True)[:top_k]

        return reranked_docs

    def extract_sentences_from_docs(self, docs):
        """
        Trích xuất các câu từ tài liệu

        Args:
            docs: Danh sách tài liệu

        Returns:
            Danh sách các câu với tiêu đề và điểm số [(title, sentence, score)]
        """
        all_sentences = []

        # Phân chia tài liệu thành các câu
        for doc_id, doc_text, doc_title, doc_score in docs:
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                sentences = nltk.sent_tokenize(doc_text)
            except:
                # Fallback nếu không thể sử dụng nltk
                sentences = [doc_text]

            # Thêm từng câu với tiêu đề và điểm số
            for sentence in sentences:
                if len(sentence.split()) >= 3:  # Bỏ qua câu quá ngắn
                    all_sentences.append((doc_title, sentence, doc_score))

        return all_sentences

    def select_knowledge_generatively(self, query, reranked_docs, dialogue_history=None, hyperlinked_history=None):
        """
        Giai đoạn 3a: Chọn tri thức bằng phương pháp tạo sinh (GenKS)

        Args:
            query: Câu truy vấn
            reranked_docs: Danh sách tài liệu đã xếp hạng lại
            dialogue_history: Lịch sử đối thoại
            hyperlinked_history: Lịch sử đối thoại đã bổ sung hyperlink

        Returns:
            Danh sách tri thức được chọn [(title, text, score)]
        """
        # Trích xuất các câu từ tài liệu
        knowledge_candidates = self.extract_sentences_from_docs(reranked_docs)

        if not knowledge_candidates:
            return []

        # Sử dụng bộ chọn tri thức tạo sinh
        selected_knowledge = self.generative_selector.select_knowledge(
            query=query,
            knowledge_candidates=knowledge_candidates,
            dialogue_history=dialogue_history,
            hyperlinked_history=hyperlinked_history,
            top_k=self.top_k_knowledge
        )

        return selected_knowledge

    def select_knowledge_by_similarity(self, query, reranked_docs, dialogue_history=None):
        """
        Giai đoạn 3b: Chọn tri thức dựa trên độ tương đồng

        Args:
            query: Câu truy vấn
            reranked_docs: Danh sách tài liệu đã xếp hạng lại
            dialogue_history: Lịch sử đối thoại

        Returns:
            Danh sách tri thức được chọn [(title, text, score)]
        """
        # Trích xuất các câu từ tài liệu
        all_sentences = self.extract_sentences_from_docs(reranked_docs)

        if not all_sentences:
            return []

        # Kết hợp truy vấn với lịch sử đối thoại nếu có
        if dialogue_history:
            last_n_turns = dialogue_history[-3:]  # Lấy 3 lượt cuối
            combined_query = " ".join(last_n_turns) + " " + query
        else:
            combined_query = query

        # Tính toán điểm tương đồng nếu có SentenceTransformer
        if hasattr(self, 'encoder'):
            try:
                # Mã hóa truy vấn và các câu
                query_embedding = self.encoder.encode(combined_query, convert_to_tensor=True)
                sentence_texts = [sent[1] for sent in all_sentences]
                sentence_embeddings = self.encoder.encode(sentence_texts, convert_to_tensor=True)

                # Tính độ tương đồng cosine
                similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

                # Lấy top-k câu có độ tương đồng cao nhất
                top_indices = similarities.argsort(descending=True)[:self.top_k_knowledge].tolist()

                # Lấy các câu tương ứng
                selected_knowledge = [all_sentences[idx] for idx in top_indices]

                return selected_knowledge
            except Exception as e:
                logger.warning(f"Lỗi khi tính toán độ tương đồng: {str(e)}")

        # Fallback: Sắp xếp theo điểm số tài liệu gốc
        sorted_sentences = sorted(all_sentences, key=lambda x: x[2], reverse=True)
        return sorted_sentences[:self.top_k_knowledge]

    def prepare_generation_input(self, query, selected_knowledge, dialogue_history=None):
        """
        Chuẩn bị đầu vào cho việc sinh phản hồi

        Args:
            query: Câu truy vấn
            selected_knowledge: Danh sách tri thức đã chọn
            dialogue_history: Lịch sử đối thoại

        Returns:
            Chuỗi đầu vào đã định dạng
        """
        # Lấy lịch sử đối thoại với hyperlink
        hyperlinked_history = None
        if dialogue_history and self.hyperlink_processor:
            hyperlinked_history = self.hyperlink_processor.get_hyperlinks_for_context(dialogue_history)

        # Xây dựng phần ngữ cảnh
        context_parts = []

        # Sử dụng lịch sử với hyperlink nếu có
        history_to_use = hyperlinked_history if hyperlinked_history else dialogue_history

        if history_to_use:
            for i, utterance in enumerate(history_to_use):
                speaker = "User1: " if i % 2 == 0 else "User2: "
                context_parts.append(f"{speaker}{utterance}")

        # Thêm truy vấn hiện tại
        current_speaker = "User1: " if len(context_parts) % 2 == 0 else "User2: "
        context_parts.append(f"{current_speaker}{query}")

        context_text = "\n".join(context_parts)

        # Xây dựng phần tri thức
        knowledge_parts = ["<knowledge>Thông tin tham khảo:"]

        for i, (title, text, _) in enumerate(selected_knowledge):
            if i < self.top_k_knowledge:
                # Sử dụng thẻ đánh dấu tri thức
                knowledge_parts.append(f"<k{i + 1}>[{title}] {text}</k{i + 1}>")

        knowledge_parts.append("</knowledge>")
        knowledge_text = "\n".join(knowledge_parts)

        # Kết hợp ngữ cảnh và tri thức
        input_text = f"{context_text}\n\n{knowledge_text}\n\nPhản hồi:"

        return input_text

    def generate_response(self, input_text, max_length=128, num_beams=4, temperature=0.7):
        """
        Giai đoạn 4: Sinh phản hồi

        Args:
            input_text: Chuỗi đầu vào
            max_length: Độ dài tối đa của phản hồi
            num_beams: Số lượng beam trong beam search
            temperature: Nhiệt độ ảnh hưởng đến tính đa dạng

        Returns:
            Phản hồi được sinh ra
        """
        # Mã hóa đầu vào
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        # Sinh phản hồi
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                temperature=temperature
            )

        # Giải mã đầu ra
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def calculate_perplexity(self, input_text, target_text):
        """
        Tính perplexity (PPL) của target_text dựa trên input_text

        Args:
            input_text: Chuỗi đầu vào
            target_text: Chuỗi đích cần tính PPL

        Returns:
            Giá trị PPL
        """
        # Mã hóa đầu vào
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        # Mã hóa đầu ra đích
        labels = self.tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device).input_ids

        # Tính loss
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels
            )
            loss = outputs.loss

        # Tính perplexity
        ppl = math.exp(loss.item())

        return ppl

    def process_query(self, query, dialogue_history=None, use_generative_selection=None):
        """
        Xử lý truy vấn theo quy trình đầy đủ

        Args:
            query: Câu truy vấn
            dialogue_history: Lịch sử đối thoại
            use_generative_selection: Có sử dụng phương pháp tạo sinh để chọn tri thức không

        Returns:
            Dict kết quả
        """
        # Xác định phương pháp chọn tri thức
        if use_generative_selection is None:
            use_generative_selection = self.use_generative_selection

        # Giai đoạn 1: Truy xuất tài liệu sơ bộ
        retrieved_docs = self.retrieve_documents(query, dialogue_history)
        logger.info(f"Giai đoạn 1: Đã truy xuất {len(retrieved_docs)} tài liệu")

        # Giai đoạn 2: Xếp hạng lại
        reranked_docs = self.rerank_documents(query, retrieved_docs, dialogue_history)
        logger.info(f"Giai đoạn 2: Đã xếp hạng lại lấy {len(reranked_docs)} tài liệu tốt nhất")

        # Lấy lịch sử đối thoại với hyperlink
        hyperlinked_history = None
        if dialogue_history and self.hyperlink_processor:
            hyperlinked_history = self.hyperlink_processor.get_hyperlinks_for_context(dialogue_history)

        # Giai đoạn 3: Chọn tri thức
        if use_generative_selection:
            # Phương pháp tạo sinh (GenKS)
            selected_knowledge = self.select_knowledge_generatively(
                query, reranked_docs, dialogue_history, hyperlinked_history
            )
            logger.info(f"Giai đoạn 3: Đã chọn {len(selected_knowledge)} tri thức bằng phương pháp tạo sinh")
        else:
            # Phương pháp dựa trên độ tương đồng
            selected_knowledge = self.select_knowledge_by_similarity(query, reranked_docs, dialogue_history)
            logger.info(f"Giai đoạn 3: Đã chọn {len(selected_knowledge)} tri thức dựa trên độ tương đồng")

        # Hiển thị tri thức đã chọn để debug
        for i, (title, text, score) in enumerate(selected_knowledge):
            logger.info(f"  Tri thức {i + 1}: [{title}] {text[:50]}... (score: {score:.4f})")

        # Chuẩn bị đầu vào cho việc sinh phản hồi
        input_text = self.prepare_generation_input(query, selected_knowledge, dialogue_history)

        # Giai đoạn 4: Sinh phản hồi
        response = self.generate_response(input_text)
        logger.info(f"Giai đoạn 4: Đã sinh phản hồi: {response[:50]}...")

        # Cập nhật hyperlink cho lượt đối thoại hiện tại
        current_turn_id = len(dialogue_history) if dialogue_history else 0
        if selected_knowledge and len(selected_knowledge) > 0:
            self.hyperlink_processor.add_hyperlink(
                current_turn_id,
                f"k1",  # Định danh của tri thức đầu tiên
                selected_knowledge[0][0]  # Tiêu đề của tri thức đầu tiên
            )

        # Tính PPL nếu có target_response
        ppl = None
        if 'target_response' in locals() and locals()['target_response']:
            target_response = locals()['target_response']
            ppl = self.calculate_perplexity(input_text, target_response)
            logger.info(f"PPL của phản hồi mục tiêu: {ppl:.2f}")

        return {
            "response": response,
            "retrieved_docs": retrieved_docs[:5],  # Chỉ trả về 5 tài liệu đầu tiên
            "reranked_docs": reranked_docs[:3],
            "selected_knowledge": selected_knowledge,
            "ppl": ppl,
            "hyperlinked_history": self.hyperlink_processor.get_hyperlinks_for_context(
                (dialogue_history or []) + [query]
            )
        }

    def save(self, path):
        """
        Lưu mô hình

        Args:
            path: Đường dẫn thư mục lưu mô hình
        """
        os.makedirs(path, exist_ok=True)

        # Lưu mô hình và tokenizer
        self.model.save_pretrained(os.path.join(path, "model"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))

        # Lưu cấu hình
        config = {
            "top_k_retrieval": self.top_k_retrieval,
            "top_k_rerank": self.top_k_rerank,
            "top_k_knowledge": self.top_k_knowledge,
            "retrieval_method": self.retrieval_method,
            "use_generative_selection": self.use_generative_selection
        }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

        logger.info(f"Đã lưu mô hình vào {path}")

    def load(self, path, device=None):
        """
        Tải mô hình

        Args:
            path: Đường dẫn thư mục chứa mô hình
            device: Thiết bị tính toán
        """
        if device:
            self.device = device

        # Tải mô hình và tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            os.path.join(path, "model")
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(path, "tokenizer")
        )

        # Tải cấu hình
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)

        self.top_k_retrieval = config.get("top_k_retrieval", self.top_k_retrieval)
        self.top_k_rerank = config.get("top_k_rerank", self.top_k_rerank)
        self.top_k_knowledge = config.get("top_k_knowledge", self.top_k_knowledge)
        self.retrieval_method = config.get("retrieval_method", self.retrieval_method)
        self.use_generative_selection = config.get("use_generative_selection", self.use_generative_selection)

        # Cập nhật bộ chọn tri thức tạo sinh
        self.generative_selector = GenerativeKnowledgeSelector(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

        logger.info(f"Đã tải mô hình từ {path}")


def evaluate_enhanced_genks(
        model,
        eval_data,
        output_file=None,
        batch_size=8,
        calculate_ppl=True
):
    """
    Đánh giá mô hình GenKS cải tiến

    Args:
        model: Mô hình GenKS cải tiến
        eval_data: Dữ liệu đánh giá
        output_file: File lưu kết quả
        batch_size: Kích thước batch
        calculate_ppl: Có tính PPL hay không

    Returns:
        Dict kết quả đánh giá
    """
    logger.info(f"Đang đánh giá mô hình với {len(eval_data)} mẫu")

    # Chuẩn bị tập dữ liệu đánh giá
    eval_dataset = ImprovedMultiSpanGENKSData(
        eval_data,
        model.tokenizer,
        context_len=256,
        knowledge_len=64,
        max_length=1024,
        test=True,
        top_k_knowledge=model.top_k_knowledge,
        add_hyperlink=True
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=eval_dataset.collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Đặt mô hình ở chế độ đánh giá
    model.model.eval()

    # Chuẩn bị biến lưu trữ kết quả
    output_texts = []
    target_texts = []
    ppls = []

    # Tiến hành đánh giá
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            # Chuyển batch sang thiết bị tính toán
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Sinh phản hồi
            outputs = model.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=512,
                num_beams=4,
                no_repeat_ngram_size=3,
                temperature=0.7
            )

            # Giải mã đầu ra và đích
            for i in range(outputs.size(0)):
                # Giải mã phản hồi sinh ra
                output_text = model.tokenizer.decode(outputs[i], skip_special_tokens=True)
                output_texts.append(output_text)

                # Giải mã phản hồi đích
                target_ids = batch['labels'][i].clone()
                target_ids[target_ids == -100] = model.tokenizer.pad_token_id
                target_text = model.tokenizer.decode(target_ids, skip_special_tokens=True)
                target_texts.append(target_text)

                # Tính PPL nếu cần
                if calculate_ppl:
                    # Lấy đầu vào
                    input_ids = batch['input_ids'][i].clone()
                    input_text = model.tokenizer.decode(input_ids, skip_special_tokens=False)

                    # Tính PPL
                    ppl = model.calculate_perplexity(input_text, target_text)
                    ppls.append(ppl)

    # Tính toán các metrics
    # Chuẩn bị dữ liệu
    refs = [[ref.lower().split()] for ref in target_texts]
    hyps = [hyp.lower().split() for hyp in output_texts]

    # Tính BLEU
    smoothie = SmoothingFunction().method1
    bleu1 = sum([sentence_bleu([ref[0]], hyp, weights=(1, 0, 0, 0), smoothing_function=smoothie)
                 for ref, hyp in zip(refs, hyps)]) / len(refs)
    bleu4 = sum([sentence_bleu([ref[0]], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
                 for ref, hyp in zip(refs, hyps)]) / len(refs)

    # Tính ROUGE
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(
            [' '.join(hyp) for hyp in hyps],
            [' '.join(ref[0]) for ref in refs],
            avg=True
        )
    except Exception as e:
        logger.warning(f"Lỗi khi tính ROUGE: {str(e)}")
        rouge_scores = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}

    # Tính Knowledge F1 (KF1)
    def f1_score(prediction, ground_truth):
        """Tính F1 giữa dự đoán và ground truth"""
        prediction_tokens = prediction.split()
        ground_truth_tokens = ground_truth.split()
        common = set(prediction_tokens) & set(ground_truth_tokens)
        num_same = len(common)

        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    kf1 = sum([f1_score(' '.join(hyp), ' '.join(ref[0]))
               for hyp, ref in zip(hyps, refs)]) / len(refs)

    # Tính trung bình PPL
    avg_ppl = sum(ppls) / len(ppls) if ppls else None

    # Tổng hợp kết quả
    results = {
        'bleu1': bleu1 * 100,
        'bleu4': bleu4 * 100,
        'rouge1': rouge_scores['rouge-1']['f'] * 100,
        'rouge2': rouge_scores['rouge-2']['f'] * 100,
        'rougeL': rouge_scores['rouge-l']['f'] * 100,
        'kf1': kf1 * 100,
        'ppl': avg_ppl
    }

    # Hiển thị kết quả
    logger.info("Kết quả đánh giá:")
    for metric, value in results.items():
        if value is not None:
            logger.info(f"{metric}: {value:.2f}")

    # Lưu kết quả nếu cần
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Lưu mẫu phản hồi
        sample_file = output_file.replace('.json', '_samples.txt')
        with open(sample_file, 'w') as f:
            for i, (pred, ref) in enumerate(zip(output_texts[:20], target_texts[:20])):  # Lưu 20 mẫu đầu tiên
                f.write(f"Mẫu {i + 1}:\n")
                f.write(f"Dự đoán: {pred}\n")
                f.write(f"Tham chiếu: {ref}\n")
                f.write("-" * 80 + "\n")

    return results


def train_enhanced_genks(
        model,
        train_data,
        eval_data=None,
        output_dir='/kaggle/working/ckpt/enhanced_genks',
        epochs=3,
        batch_size=8,
        accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=500,
        max_grad_norm=1.0
):
    """
    Huấn luyện mô hình GenKS cải tiến

    Args:
        model: Mô hình GenKS cải tiến
        train_data: Dữ liệu huấn luyện
        eval_data: Dữ liệu đánh giá
        output_dir: Thư mục lưu mô hình
        epochs: Số epochs
        batch_size: Kích thước batch
        accumulation_steps: Số bước tích lũy gradient
        learning_rate: Tốc độ học
        warmup_steps: Số bước làm nóng
        max_grad_norm: Ngưỡng cắt gradient

    Returns:
        Mô hình đã huấn luyện
    """
    # Khởi tạo accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps
    )

    logger.info(f"Đang huấn luyện mô hình với {len(train_data)} mẫu trong {epochs} epochs")

    # Chuẩn bị tập dữ liệu
    train_dataset = ImprovedMultiSpanGENKSData(
        train_data,
        model.tokenizer,
        context_len=512,
        knowledge_len=128,
        max_length=1024,
        test=False,
        top_k_knowledge=model.top_k_knowledge,
        add_hyperlink=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # Tổng số bước huấn luyện
    total_steps = len(train_dataloader) * epochs // accumulation_steps

    # Khởi tạo optimizer và scheduler
    optimizer = AdamW(
        model.model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Scheduler với warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Chuẩn bị với accelerator
    model.model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model.model, optimizer, train_dataloader, scheduler
    )

    # Vòng lặp huấn luyện
    best_eval_result = 0

    for epoch in range(epochs):
        # Đảm bảo tất cả các tiến trình đồng bộ
        accelerator.wait_for_everyone()

        logger.info(f"Epoch {epoch + 1}/{epochs}")

        # Đặt mô hình ở chế độ huấn luyện
        model.model.train()

        # Thanh tiến trình
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
        running_loss = 0

        # Duyệt qua các batch
        for step, batch in enumerate(progress_bar):
            # Tích lũy gradient
            with accelerator.accumulate(model.model):
                # Forward pass
                outputs = model.model(**batch)
                loss = outputs.loss

                # Backward pass
                accelerator.backward(loss)

                # Cắt gradient để tránh bùng nổ
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.model.parameters(), max_grad_norm)

                # Cập nhật tham số
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Cập nhật thanh tiến trình
            running_loss += loss.item()
            progress_bar.set_description(f"Loss: {running_loss / (step + 1):.4f}")

        # Lưu mô hình sau mỗi epoch
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model.model)
            unwrapped_model.save_pretrained(f"{output_dir}/epoch_{epoch + 1}")
            model.tokenizer.save_pretrained(f"{output_dir}/epoch_{epoch + 1}")

        # Đánh giá nếu có dữ liệu đánh giá
        if eval_data:
            # Unwrap mô hình
            model.model = accelerator.unwrap_model(model.model)

            # Tiến hành đánh giá
            eval_results = evaluate_enhanced_genks(
                model=model,
                eval_data=eval_data,
                output_file=f"{output_dir}/eval_results_epoch_{epoch + 1}.json",
                batch_size=batch_size,
                calculate_ppl=True
            )

            # Kiểm tra xem đây có phải mô hình tốt nhất hay không
            current_metric = eval_results.get('kf1', 0) + eval_results.get('bleu4', 0)

            if current_metric > best_eval_result:
                best_eval_result = current_metric

                # Lưu mô hình tốt nhất
                if accelerator.is_main_process:
                    logger.info(f"Tìm thấy mô hình tốt nhất ở epoch {epoch + 1}")
                    model.save(f"{output_dir}/best_model")

            # Wrap lại mô hình
            model.model = accelerator.prepare(model.model)

    # Lưu mô hình cuối cùng
    if accelerator.is_main_process:
        model.save(f"{output_dir}/final_model")

    return model


def main():
    """Hàm chính để chạy quá trình huấn luyện và đánh giá"""
    # Thiết lập logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Khởi tạo mô hình
    model = EnhancedGenKSWithRAG(
        model_name='facebook/bart-base',
        retriever_model_name='facebook/dpr-ctx_encoder-single-nq-base',
        query_encoder_name='facebook/dpr-question_encoder-single-nq-base',
        ranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
        top_k_retrieval=100,
        top_k_rerank=20,
        top_k_knowledge=3,
        retrieval_method='bm25',
        use_generative_selection=True,
        device='cuda'
    )

    # Tải dữ liệu từ file JSON
    train_data = json.load(open('/kaggle/input/wizard/train30.json'))
    valid_data = json.load(open('/kaggle/input/wizard/valid_seen.json'))
    test_seen_data = json.load(open('/kaggle/input/wizard/test_seen.json'))
    test_unseen_data = json.load(open('/kaggle/input/wizard/test_unseen.json'))

    # Xây dựng corpus từ dữ liệu huấn luyện
    corpus = []
    for i, example in enumerate(train_data):
        if 'knowledge' in example:
            for title, sentences in example['knowledge'].items():
                for j, sentence in enumerate(sentences):
                    doc_id = f"doc_{i}_{j}"
                    corpus.append((doc_id, sentence, title))

    # Xây dựng chỉ mục cho corpus
    model.build_corpus_index(corpus, cache_path='/kaggle/working/ckpt/embeddings_cache.pt')

    # Huấn luyện mô hình
    model = train_enhanced_genks(
        model=model,
        train_data=train_data,
        eval_data=valid_data,
        output_dir='/kaggle/working/ckpt/enhanced_genks',
        epochs=5,
        batch_size=8,
        accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=1000
    )

    # Đánh giá mô hình trên tập test seen
    results_seen = evaluate_enhanced_genks(
        model=model,
        eval_data=test_seen_data,
        output_file='/kaggle/working/ckpt/enhanced_genks/results_seen.json',
        batch_size=16
    )

    # Đánh giá mô hình trên tập test unseen
    results_unseen = evaluate_enhanced_genks(
        model=model,
        eval_data=test_unseen_data,
        output_file='/kaggle/working/ckpt/enhanced_genks/results_unseen.json',
        batch_size=16
    )

    # Hiển thị kết quả cuối cùng
    logger.info("\nKết quả trên tập test seen:")
    for metric, value in results_seen.items():
        logger.info(f"{metric}: {value:.2f}")

    logger.info("\nKết quả trên tập test unseen:")
    for metric, value in results_unseen.items():
        logger.info(f"{metric}: {value:.2f}")


if __name__ == '__main__':
    main()