import logging
import os
import json
import re
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DPRContextEncoder,
    DPRQuestionEncoder,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import math
import time
from datetime import datetime, timedelta
import psutil
from typing import Dict, List, Tuple, Optional, Any, Union

# Setup logging với format chi tiết
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Regular expressions để normalize text
RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


# ================================================================================
# PHẦN 1: UTILITY FUNCTIONS VÀ HELPER CLASSES
# ================================================================================

def normalize_answer(s: str) -> str:
    """
    Chuẩn hóa text để so sánh công bằng giữa prediction và ground truth.
    Loại bỏ articles, dấu câu, và chuẩn hóa khoảng trắng.
    """

    def remove_articles(text):
        return RE_ART.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return RE_PUNC.sub(' ', text)

    def lower(text):
        return text.lower()

    # Áp dụng theo thứ tự để chuẩn hóa hoàn toàn
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_unigram_f1(prediction: str, ground_truth: str) -> float:
    """
    Tính Unigram F1 score - đo lường overlap về từ vựng giữa prediction và ground truth.
    Đây là metric quan trọng để đánh giá chất lượng response.
    """
    # Normalize cả hai texts trước khi so sánh
    prediction_normalized = normalize_answer(prediction)
    ground_truth_normalized = normalize_answer(ground_truth)

    prediction_tokens = prediction_normalized.split()
    ground_truth_tokens = ground_truth_normalized.split()

    # Edge cases: một hoặc cả hai đều rỗng
    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0

    # Đếm tần suất từ để tính chính xác
    pred_counter = Counter(prediction_tokens)
    truth_counter = Counter(ground_truth_tokens)

    # Tìm từ chung và đếm số lần xuất hiện
    common_tokens = 0
    for token in pred_counter:
        if token in truth_counter:
            common_tokens += min(pred_counter[token], truth_counter[token])

    # Tính precision, recall và F1
    precision = common_tokens / sum(pred_counter.values()) if sum(pred_counter.values()) > 0 else 0
    recall = common_tokens / sum(truth_counter.values()) if sum(truth_counter.values()) > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def calculate_knowledge_f1(prediction: str, gold_knowledge: str) -> float:
    """
    Tính Knowledge F1 - metric quan trọng để phát hiện hallucination.
    So sánh response với gold knowledge để xem model có dùng đúng tri thức không.

    KF1 cao + F1 thấp: Model dùng đúng knowledge nhưng diễn đạt khác
    KF1 thấp + F1 cao: Model có thể đang hallucinate (tự tạo thông tin)
    """
    if not gold_knowledge or gold_knowledge == "no_passages_used":
        return 0.0

    # Normalize và tokenize
    prediction_normalized = normalize_answer(prediction)
    gold_knowledge_normalized = normalize_answer(gold_knowledge)

    prediction_tokens = prediction_normalized.split()
    knowledge_tokens = gold_knowledge_normalized.split()

    if len(prediction_tokens) == 0 and len(knowledge_tokens) == 0:
        return 1.0
    if len(prediction_tokens) == 0 or len(knowledge_tokens) == 0:
        return 0.0

    # Tính overlap với Counter để handle duplicates
    pred_counter = Counter(prediction_tokens)
    knowledge_counter = Counter(knowledge_tokens)

    common = pred_counter & knowledge_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    # Precision: Bao nhiêu % response từ gold knowledge
    precision = num_same / sum(pred_counter.values())

    # Recall: Bao nhiêu % gold knowledge được mention
    recall = num_same / sum(knowledge_counter.values())

    # F1 score
    kf1 = (2 * precision * recall) / (precision + recall)
    return kf1


class GPUMonitor:
    """
    Monitor GPU và system resources trong quá trình training/evaluation.
    Giúp track memory usage và phát hiện memory leaks.
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.monitoring_data = []
        self.peak_memory = 0
        self.start_time = None
        self.gpu_available = torch.cuda.is_available()

        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            self.gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU Monitor initialized: {self.gpu_count} GPU(s) - {self.gpu_name}")
        else:
            logger.warning("No GPU found, monitoring CPU/RAM only")

    def get_gpu_stats(self) -> Dict:
        """Thu thập thông tin GPU và system hiện tại"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_used_gb': psutil.virtual_memory().used / (1024 ** 3),
            'ram_percent': psutil.virtual_memory().percent
        }

        if self.gpu_available:
            # GPU memory stats
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

            stats['gpu_memory_allocated_gb'] = allocated
            stats['gpu_memory_reserved_gb'] = reserved
            stats['gpu_memory_total_gb'] = total
            stats['gpu_memory_free_gb'] = total - allocated
            stats['gpu_memory_percent'] = (allocated / total) * 100

            # Update peak memory
            if allocated > self.peak_memory:
                self.peak_memory = allocated
            stats['peak_memory_gb'] = self.peak_memory

        return stats

    def start_monitoring(self):
        """Bắt đầu monitoring session"""
        self.start_time = time.time()
        initial_stats = self.get_gpu_stats()
        initial_stats['event'] = 'start_monitoring'
        self.monitoring_data.append(initial_stats)
        logger.info(f"Started GPU/System monitoring at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def log_stats(self, event_name="checkpoint", extra_info=None):
        """Log stats tại một thời điểm cụ thể"""
        stats = self.get_gpu_stats()
        stats['event'] = event_name

        if self.start_time:
            elapsed = time.time() - self.start_time
            stats['elapsed_time_seconds'] = elapsed
            stats['elapsed_time_formatted'] = str(timedelta(seconds=int(elapsed)))

        if extra_info:
            stats.update(extra_info)

        self.monitoring_data.append(stats)

        if self.gpu_available:
            logger.info(
                f"[{event_name}] GPU Memory: {stats['gpu_memory_allocated_gb']:.2f}/{stats['gpu_memory_total_gb']:.2f} GB "
                f"({stats['gpu_memory_percent']:.1f}%) | Peak: {self.peak_memory:.2f} GB"
            )


class HyperlinkProcessor:
    """
    Xử lý hyperlinks giữa dialogue và knowledge.
    """

    def __init__(self):
        self.dialogue_knowledge_mapping = {}

    def add_hyperlink(self, turn_id: int, knowledge_id: str, title: str):
        """Thêm mapping giữa dialogue turn và knowledge"""
        self.dialogue_knowledge_mapping[turn_id] = (knowledge_id, title)

    def get_hyperlinks_for_context(self, dialogue_history: List[str]) -> List[str]:
        """Tạo hyperlinked version của dialogue history"""
        hyperlinked_history = []

        for i, utterance in enumerate(dialogue_history):
            if i in self.dialogue_knowledge_mapping:
                knowledge_id, title = self.dialogue_knowledge_mapping[i]
                # Format: [title]<knowledge_id> utterance
                hyperlinked_utterance = f"[{title}]<{knowledge_id}> {utterance}"
                hyperlinked_history.append(hyperlinked_utterance)
            else:
                hyperlinked_history.append(utterance)

        return hyperlinked_history


# ================================================================================
# PHẦN 2: KNOWLEDGE SELECTOR
# ================================================================================

class GenerativeKnowledgeSelector:
    """
    Bộ chọn tri thức sử dụng phương pháp generative.
    Model sinh ra identifiers (<k1>, <k2>...) để chọn knowledge phù hợp.
    """

    def __init__(self, model, tokenizer, max_knowledge_candidates=20, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.max_knowledge_candidates = max_knowledge_candidates
        self.device = device

    def prepare_input_for_generation(
            self,
            query: str,
            dialogue_history: List[str],
            knowledge_candidates: List[Tuple[str, str, float]],
            hyperlinked_history: Optional[List[str]] = None
    ) -> str:
        """
        Chuẩn bị input cho model selection.
        Format: Context + Query + Knowledge Candidates → Selection
        """
        context_parts = []
        history_to_use = hyperlinked_history if hyperlinked_history else dialogue_history

        # Format dialogue history với speaker labels
        if history_to_use:
            for i, utterance in enumerate(history_to_use):
                speaker = "User1: " if i % 2 == 0 else "User2: "
                context_parts.append(f"{speaker}{utterance}")

        # Add current query
        current_speaker = "User1: " if len(context_parts) % 2 == 0 else "User2: "
        context_parts.append(f"{current_speaker}{query}")
        context_text = "\n".join(context_parts)

        # Format knowledge candidates với identifiers
        knowledge_parts = ["Reference Information:"]
        candidates_to_use = knowledge_candidates[:self.max_knowledge_candidates]

        for i, (title, text, _) in enumerate(candidates_to_use):
            knowledge_parts.append(f"<k{i + 1}> [{title}] {text}")

        knowledge_text = "\n".join(knowledge_parts)

        # Final input format
        input_text = f"{context_text}\n\n{knowledge_text}\n\nKnowledge Selection:"

        return input_text

    def select_knowledge(
            self,
            query: str,
            knowledge_candidates: List[Tuple[str, str, float]],
            dialogue_history: Optional[List[str]] = None,
            hyperlinked_history: Optional[List[str]] = None,
            top_k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Chọn top-k knowledge pieces phù hợp nhất.
        Model sinh ra sequence của identifiers (<k1>, <k2>...) để chọn.
        """
        if not knowledge_candidates:
            return []

        # Prepare input
        input_text = self.prepare_input_for_generation(
            query, dialogue_history or [], knowledge_candidates, hyperlinked_history
        )

        # Tokenize và generate
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=20,  # Chỉ cần generate identifiers
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        logger.debug(f"Generated selection: {generated_text}")

        # Parse identifiers từ generated text
        selected_knowledge = []
        pattern = r'<k(\d+)>'
        matches = re.findall(pattern, generated_text)

        for match in matches:
            idx = int(match) - 1
            if idx < len(knowledge_candidates):
                selected_knowledge.append(knowledge_candidates[idx])
                if len(selected_knowledge) >= top_k:
                    break

        # Nếu chưa đủ, fill với highest scored candidates
        if len(selected_knowledge) < top_k:
            remaining_candidates = [
                c for c in knowledge_candidates
                if all(c[1] != k[1] for k in selected_knowledge)
            ]
            remaining_candidates.sort(key=lambda x: x[2], reverse=True)
            selected_knowledge.extend(remaining_candidates[:top_k - len(selected_knowledge)])

        return selected_knowledge


# ================================================================================
# PHẦN 3: DATASET CLASS
# ================================================================================

class RAGenKSDataset(Dataset):
    """
    Dataset cho training RA-GenKS với teacher forcing.
    Trong training, sử dụng gold knowledge thay vì retrieved knowledge.
    """

    def __init__(
            self,
            data: List[Dict],
            tokenizer,
            max_context_length: int = 512,
            max_knowledge_length: int = 256,
            max_response_length: int = 128,
            test: bool = False,
            top_k_knowledge: int = 3,
            add_hyperlink: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_knowledge_length = max_knowledge_length
        self.max_response_length = max_response_length
        self.test = test
        self.top_k_knowledge = top_k_knowledge
        self.add_hyperlink = add_hyperlink
        self.hyperlink_processor = HyperlinkProcessor()

        # Add special tokens nếu chưa có
        special_tokens = ['<knowledge>', '</knowledge>']
        for i in range(1, top_k_knowledge + 1):
            special_tokens.extend([f'<k{i}>', f'</k{i}>'])

        tokens_to_add = [t for t in special_tokens if t not in tokenizer.get_vocab()]
        if tokens_to_add:
            tokenizer.add_tokens(tokens_to_add)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Chuẩn bị một training example.
        Trong training, dùng gold knowledge (teacher forcing).
        """
        example = self.data[index]

        # Build context từ dialogue history
        context_parts = []

        # Add topic nếu có
        if 'chosen_topic' in example and example['chosen_topic']:
            context_parts.append(f"Topic: {example['chosen_topic']}")

        # Process dialogue history
        dialogue_history = []
        if 'context' in example:
            for turn in example['context']:
                text = turn.get('text', '')
                dialogue_history.append(text)

        # Format dialogue với speaker labels
        for i, utterance in enumerate(dialogue_history):
            speaker = "User1: " if i % 2 == 0 else "User2: "
            context_parts.append(f"{speaker}{utterance}")

        # Add current query
        query = example.get('query', '') or example.get('text', '')
        if query:
            current_speaker = "User1: " if len(dialogue_history) % 2 == 0 else "User2: "
            context_parts.append(f"{current_speaker}{query}")

        context_text = "\n".join(context_parts)

        # Prepare knowledge (trong training dùng gold knowledge)
        selected_knowledge = []

        # Gold knowledge từ dataset
        gold_knowledge = example.get('checked_sentence', '')
        gold_title = example.get('title', '')

        if gold_knowledge and gold_knowledge != 'no_passages_used':
            selected_knowledge.append((gold_title, gold_knowledge, 1.0))

        # Thêm additional knowledge nếu cần (để đủ top_k)
        if 'knowledge' in example:
            for title, sentences in example['knowledge'].items():
                for sentence in sentences:
                    if sentence != gold_knowledge:  # Tránh duplicate
                        selected_knowledge.append((title, sentence, 0.5))
                        if len(selected_knowledge) >= self.top_k_knowledge:
                            break
                if len(selected_knowledge) >= self.top_k_knowledge:
                    break

        # Format input với knowledge
        if selected_knowledge:
            knowledge_parts = ["<knowledge>Reference Information:"]
            for i, (title, text, _) in enumerate(selected_knowledge[:self.top_k_knowledge]):
                knowledge_parts.append(f"<k{i + 1}>[{title}] {text}</k{i + 1}>")
            knowledge_parts.append("</knowledge>")
            knowledge_text = "\n".join(knowledge_parts)
            input_sequence = f"{context_text}\n\n{knowledge_text}\n\nResponse:"
        else:
            input_sequence = f"{context_text}\n\nResponse:"

        # Target response
        target_response = example.get('response', '') or (
            example.get('labels', [''])[0] if 'labels' in example else ''
        )

        # Tokenize input
        inputs = self.tokenizer(
            input_sequence,
            truncation=True,
            max_length=self.max_context_length + self.max_knowledge_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_response,
                truncation=True,
                max_length=self.max_response_length,
                padding='max_length',
                return_tensors='pt'
            ).input_ids

        # Replace padding token id với -100 cho loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': labels.squeeze()
        }


# ================================================================================
# PHẦN 4: MAIN MODEL - ENHANCED RA-GENKS
# ================================================================================

class EnhancedRAGenKS:
    """
    Mô hình RA-GenKS hoàn chỉnh với 4 giai đoạn pipeline.

    Kiến trúc:
    1. Retrieval: BM25/DPR/Sentence-BERT
    2. Reranking: Cross-Encoder
    3. Knowledge Selection: Generative/Similarity-based
    4. Response Generation: BART/T5
    """

    def __init__(
            self,
            model_name: str = 'facebook/bart-base',
            retriever_model_name: str = 'facebook/dpr-ctx_encoder-single-nq-base',
            query_encoder_name: str = 'facebook/dpr-question_encoder-single-nq-base',
            ranker_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            embed_dim: int = 768,
            top_k_retrieval: int = 100,
            top_k_rerank: int = 20,
            top_k_knowledge: int = 3,
            retrieval_method: str = 'bm25',
            use_generative_selection: bool = True,
            device: str = 'cuda',
            cache_dir: Optional[str] = None
    ):
        """
        Khởi tạo mô hình với các components.

        Args:
            model_name: Pretrained model cho generation
            retrieval_method: 'bm25', 'dpr', hoặc 'sentence_bert'
            use_generative_selection: True để dùng generative selection
            top_k_*: Số lượng candidates ở mỗi stage
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.top_k_knowledge = top_k_knowledge
        self.retrieval_method = retrieval_method.lower()
        self.use_generative_selection = use_generative_selection

        logger.info(f"Initializing Enhanced RA-GenKS on {self.device}")
        logger.info(
            f"Configuration: retrieval={retrieval_method}, selection={'generative' if use_generative_selection else 'similarity'}")

        # Initialize generation model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)

        # Add special tokens for knowledge marking
        special_tokens = ['<knowledge>', '</knowledge>']
        for i in range(1, top_k_knowledge + 1):
            special_tokens.extend([f'<k{i}>', f'</k{i}>'])

        num_added = self.tokenizer.add_tokens(special_tokens)
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added {num_added} special tokens to vocabulary")

        # Initialize retrieval components
        self.corpus = []
        self.corpus_embeddings = None
        self.faiss_index = None

        if retrieval_method == 'dpr':
            # Dense Passage Retrieval
            self.ctx_encoder = DPRContextEncoder.from_pretrained(
                retriever_model_name, cache_dir=cache_dir
            ).to(self.device)
            self.query_encoder = DPRQuestionEncoder.from_pretrained(
                query_encoder_name, cache_dir=cache_dir
            ).to(self.device)
            logger.info("Initialized DPR encoders")

        elif retrieval_method == 'sentence_bert':
            # Sentence-BERT for retrieval
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
            logger.info("Initialized Sentence-BERT encoder")

        else:  # BM25
            from rank_bm25 import BM25Okapi
            self.bm25 = None  # Will be initialized with corpus
            logger.info("Will use BM25 for retrieval")

        # Initialize reranker
        try:
            self.ranker = CrossEncoder(ranker_model_name)
            logger.info(f"Initialized CrossEncoder reranker: {ranker_model_name}")
        except Exception as e:
            self.ranker = None
            logger.warning(f"CrossEncoder not available: {e}. Will use retrieval scores for ranking.")

        # Initialize knowledge selector
        self.generative_selector = GenerativeKnowledgeSelector(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

        # Initialize hyperlink processor
        self.hyperlink_processor = HyperlinkProcessor()

        logger.info("Enhanced RA-GenKS initialization complete")

    def build_corpus_index(self, corpus_data: List[Dict], cache_path: Optional[str] = None):
        """
        Xây dựng index cho retrieval từ corpus data.
        Corpus có thể từ training data hoặc external knowledge base.
        """
        logger.info(f"Building corpus index from {len(corpus_data)} documents...")

        self.corpus = []

        # Extract tất cả knowledge pieces từ data
        for item in tqdm(corpus_data, desc="Processing corpus"):
            if 'knowledge' in item:
                for title, sentences in item['knowledge'].items():
                    for sentence in sentences:
                        if len(sentence.split()) >= 5:  # Filter quá ngắn
                            self.corpus.append({
                                'id': len(self.corpus),
                                'title': title,
                                'text': sentence
                            })

        logger.info(f"Corpus contains {len(self.corpus)} knowledge pieces")

        # Build index dựa trên retrieval method
        if self.retrieval_method == 'bm25':
            # BM25 index
            from rank_bm25 import BM25Okapi

            # Tokenize corpus cho BM25
            tokenized_corpus = [doc['text'].lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built successfully")

        elif self.retrieval_method in ['dpr', 'sentence_bert']:
            # Dense retrieval với FAISS
            import faiss

            # Check cache để tránh re-compute embeddings
            if cache_path and os.path.exists(cache_path):
                logger.info(f"Loading embeddings from cache: {cache_path}")
                self.corpus_embeddings = torch.load(cache_path)
            else:
                logger.info("Computing embeddings for corpus...")
                texts = [doc['text'] for doc in self.corpus]

                if self.retrieval_method == 'dpr':
                    # Encode với DPR context encoder
                    batch_size = 32
                    embeddings = []

                    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with DPR"):
                        batch_texts = texts[i:i + batch_size]
                        inputs = self.ctx_encoder.ctx_encoder.tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt'
                        ).to(self.device)

                        with torch.no_grad():
                            batch_embeddings = self.ctx_encoder(**inputs).pooler_output
                        embeddings.append(batch_embeddings.cpu())

                    self.corpus_embeddings = torch.cat(embeddings, dim=0).numpy()

                else:  # sentence_bert
                    self.corpus_embeddings = self.encoder.encode(
                        texts,
                        convert_to_tensor=False,
                        show_progress_bar=True,
                        batch_size=32
                    )

                # Save embeddings to cache
                if cache_path:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(self.corpus_embeddings, cache_path)
                    logger.info(f"Saved embeddings to cache: {cache_path}")

            # Build FAISS index
            dim = self.corpus_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product for similarity

            # Normalize vectors cho cosine similarity
            faiss.normalize_L2(self.corpus_embeddings)
            self.faiss_index.add(self.corpus_embeddings)

            logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")

    def retrieve_documents(
            self,
            query: str,
            dialogue_history: Optional[List[str]] = None,
            top_k: Optional[int] = None
    ) -> List[Tuple[int, str, str, float]]:
        """
        Giai đoạn 1: RETRIEVAL
        Truy xuất top-k documents từ corpus dựa trên query và context.

        Returns:
            List of (doc_id, text, title, score)
        """
        if top_k is None:
            top_k = self.top_k_retrieval

        # Combine query với recent dialogue context
        if dialogue_history:
            # Lấy 3 turns gần nhất cho context
            recent_context = " ".join(dialogue_history[-3:])
            combined_query = f"{recent_context} {query}"
        else:
            combined_query = query

        retrieved_docs = []

        if self.retrieval_method == 'bm25':
            # BM25 retrieval
            if self.bm25 is None:
                logger.error("BM25 index not built. Call build_corpus_index first.")
                return []

            # Tokenize query cho BM25
            tokenized_query = combined_query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-k documents
            top_indices = np.argsort(scores)[::-1][:top_k]

            for idx in top_indices:
                if idx < len(self.corpus):
                    doc = self.corpus[idx]
                    retrieved_docs.append((
                        doc['id'],
                        doc['text'],
                        doc['title'],
                        float(scores[idx])
                    ))

        elif self.retrieval_method in ['dpr', 'sentence_bert']:
            # Dense retrieval với FAISS
            import faiss

            if self.faiss_index is None:
                logger.error("FAISS index not built. Call build_corpus_index first.")
                return []

            # Encode query
            if self.retrieval_method == 'dpr':
                # Use DPR query encoder
                inputs = self.query_encoder.question_encoder.tokenizer(
                    combined_query,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    query_embedding = self.query_encoder(**inputs).pooler_output
                query_embedding = query_embedding.cpu().numpy()

            else:  # sentence_bert
                query_embedding = self.encoder.encode(
                    [combined_query],
                    convert_to_tensor=False
                )

            # Normalize và search
            faiss.normalize_L2(query_embedding)
            scores, indices = self.faiss_index.search(query_embedding, top_k)

            # Convert results
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.corpus):
                    doc = self.corpus[idx]
                    retrieved_docs.append((
                        doc['id'],
                        doc['text'],
                        doc['title'],
                        float(score)
                    ))

        logger.debug(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs

    def rerank_documents(
            self,
            query: str,
            docs: List[Tuple[int, str, str, float]],
            dialogue_history: Optional[List[str]] = None,
            top_k: Optional[int] = None
    ) -> List[Tuple[int, str, str, float]]:
        """
        Giai đoạn 2: RERANKING
        Xếp hạng lại documents với Cross-Encoder để tìm relevant nhất.

        Returns:
            Reranked list of (doc_id, text, title, score)
        """
        if top_k is None:
            top_k = self.top_k_rerank

        if not docs:
            return []

        # Combine query với context cho reranking
        if dialogue_history:
            recent_context = " ".join(dialogue_history[-3:])
            combined_query = f"{recent_context} {query}"
        else:
            combined_query = query

        if self.ranker:
            # Use Cross-Encoder để rerank
            logger.debug(f"Reranking {len(docs)} documents with CrossEncoder")

            # Prepare pairs cho Cross-Encoder
            pairs = [(combined_query, doc[1]) for doc in docs]

            # Get scores từ Cross-Encoder
            try:
                ce_scores = self.ranker.predict(pairs)

                # Combine Cross-Encoder scores với retrieval scores
                # Weight: 70% Cross-Encoder, 30% retrieval
                reranked_docs = []
                for i, doc in enumerate(docs):
                    combined_score = 0.7 * float(ce_scores[i]) + 0.3 * doc[3]
                    reranked_docs.append((doc[0], doc[1], doc[2], combined_score))

                # Sort by combined score
                reranked_docs = sorted(reranked_docs, key=lambda x: x[3], reverse=True)[:top_k]

            except Exception as e:
                logger.warning(f"Cross-Encoder reranking failed: {e}. Using retrieval scores.")
                reranked_docs = sorted(docs, key=lambda x: x[3], reverse=True)[:top_k]
        else:
            # Fallback: use retrieval scores
            logger.debug("No Cross-Encoder available, using retrieval scores for ranking")
            reranked_docs = sorted(docs, key=lambda x: x[3], reverse=True)[:top_k]

        logger.debug(f"Reranked to top {len(reranked_docs)} documents")
        return reranked_docs

    def select_knowledge_generatively(
            self,
            query: str,
            reranked_docs: List[Tuple[int, str, str, float]],
            dialogue_history: Optional[List[str]] = None,
            hyperlinked_history: Optional[List[str]] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Giai đoạn 3a: KNOWLEDGE SELECTION (Generative)
        Sử dụng model để sinh identifiers chọn knowledge pieces.

        Returns:
            List of (title, text, score)
        """
        # Extract sentences từ documents
        knowledge_candidates = []

        for doc_id, doc_text, doc_title, doc_score in reranked_docs:
            # Simple sentence splitting (có thể dùng NLTK cho tốt hơn)
            sentences = doc_text.split('. ')

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) >= 5:  # Filter sentences quá ngắn
                    knowledge_candidates.append((doc_title, sentence, doc_score))

        if not knowledge_candidates:
            logger.warning("No knowledge candidates found for selection")
            return []

        logger.debug(f"Selecting from {len(knowledge_candidates)} knowledge candidates")

        # Use generative selector
        selected = self.generative_selector.select_knowledge(
            query=query,
            knowledge_candidates=knowledge_candidates,
            dialogue_history=dialogue_history,
            hyperlinked_history=hyperlinked_history,
            top_k=self.top_k_knowledge
        )

        return selected

    def select_knowledge_by_similarity(
            self,
            query: str,
            reranked_docs: List[Tuple[int, str, str, float]],
            dialogue_history: Optional[List[str]] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Giai đoạn 3b: KNOWLEDGE SELECTION (Similarity-based)
        Chọn knowledge dựa trên similarity với query và context.

        Returns:
            List of (title, text, score)
        """
        # Extract sentences
        all_sentences = []

        for doc_id, doc_text, doc_title, doc_score in reranked_docs:
            sentences = doc_text.split('. ')

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) >= 5:
                    all_sentences.append((doc_title, sentence, doc_score))

        if not all_sentences:
            logger.warning("No sentences found for selection")
            return []

        # Combine query với context
        if dialogue_history:
            recent_context = " ".join(dialogue_history[-3:])
            combined_query = f"{recent_context} {query}"
        else:
            combined_query = query

        # Calculate similarity và select top-k
        if hasattr(self, 'encoder'):  # Sentence-BERT available
            logger.debug("Using Sentence-BERT for similarity-based selection")

            # Encode query và sentences
            query_embedding = self.encoder.encode(combined_query, convert_to_tensor=True)
            sentence_texts = [sent[1] for sent in all_sentences]
            sentence_embeddings = self.encoder.encode(sentence_texts, convert_to_tensor=True)

            # Calculate cosine similarities
            similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

            # Get top-k
            top_indices = similarities.argsort(descending=True)[:self.top_k_knowledge].tolist()

            selected = [all_sentences[idx] for idx in top_indices]

        else:
            # Fallback: use document scores
            logger.debug("Using document scores for selection")
            selected = sorted(all_sentences, key=lambda x: x[2], reverse=True)[:self.top_k_knowledge]

        return selected

    def prepare_generation_input(
            self,
            query: str,
            selected_knowledge: List[Tuple[str, str, float]],
            dialogue_history: Optional[List[str]] = None
    ) -> str:
        """
        Chuẩn bị input cho generation stage.
        Format: Context + Query + Selected Knowledge → Response
        """
        context_parts = []

        # Add dialogue history
        if dialogue_history:
            for i, utterance in enumerate(dialogue_history):
                speaker = "User1: " if i % 2 == 0 else "User2: "
                context_parts.append(f"{speaker}{utterance}")

        # Add current query
        current_speaker = "User1: " if len(context_parts) % 2 == 0 else "User2: "
        context_parts.append(f"{current_speaker}{query}")

        context_text = "\n".join(context_parts)

        # Add selected knowledge
        knowledge_parts = ["<knowledge>Reference Information:"]

        for i, (title, text, _) in enumerate(selected_knowledge):
            if i < self.top_k_knowledge:
                knowledge_parts.append(f"<k{i + 1}>[{title}] {text}</k{i + 1}>")

        knowledge_parts.append("</knowledge>")
        knowledge_text = "\n".join(knowledge_parts)

        # Combine all parts
        input_text = f"{context_text}\n\n{knowledge_text}\n\nResponse:"

        return input_text

    def generate_response(
            self,
            input_text: str,
            max_length: int = 128,
            num_beams: int = 4,
            temperature: float = 0.7,
            do_sample: bool = False
    ) -> str:
        """
        Giai đoạn 4: RESPONSE GENERATION
        Sinh câu trả lời dựa trên context và selected knowledge.
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                temperature=temperature,
                do_sample=do_sample,
                early_stopping=True
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def calculate_perplexity(self, input_text: str, target_text: str) -> float:
        """
        Tính perplexity của target text given input.
        Dùng để evaluate generation quality.
        """
        # Tokenize input và target
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).input_ids.to(self.device)

        # Replace padding với -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Calculate loss
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels
            )
            loss = outputs.loss

        # Perplexity = exp(loss)
        ppl = math.exp(loss.item()) if loss is not None else float('inf')

        return ppl

    def process_query(
            self,
            query: str,
            dialogue_history: Optional[List[str]] = None,
            use_generative_selection: Optional[bool] = None,
            verbose: bool = True
    ) -> Dict[str, Any]:
        """
        MAIN PIPELINE FUNCTION
        Xử lý query qua đầy đủ 4 giai đoạn của RA-GenKS.
        Đây là function quan trọng nhất cho inference và end-to-end evaluation.

        Args:
            query: User query
            dialogue_history: Previous dialogue turns
            use_generative_selection: Override default selection method
            verbose: Print detailed logs

        Returns:
            Dict containing:
            - response: Generated response
            - retrieved_docs: Documents từ retrieval
            - reranked_docs: Documents sau reranking
            - selected_knowledge: Knowledge pieces đã chọn
            - timing: Thời gian cho mỗi stage
            - scores: Các metrics
        """
        if use_generative_selection is None:
            use_generative_selection = self.use_generative_selection

        if verbose:
            logger.info("=" * 60)
            logger.info("PROCESSING QUERY THROUGH RA-GENKS PIPELINE")
            logger.info("=" * 60)
            logger.info(f"Query: {query}")
            if dialogue_history:
                logger.info(f"Context: {len(dialogue_history)} previous turns")

        timing = {}

        # STAGE 1: RETRIEVAL
        start_time = time.time()
        retrieved_docs = self.retrieve_documents(query, dialogue_history)
        timing['retrieval'] = time.time() - start_time

        if verbose:
            logger.info(f"✅ Stage 1 - Retrieval: {len(retrieved_docs)} docs in {timing['retrieval']:.2f}s")
            if retrieved_docs:
                logger.info(f"   Top doc: [{retrieved_docs[0][2]}] {retrieved_docs[0][1][:50]}...")

        # STAGE 2: RERANKING
        start_time = time.time()
        reranked_docs = self.rerank_documents(query, retrieved_docs, dialogue_history)
        timing['reranking'] = time.time() - start_time

        if verbose:
            logger.info(f"✅ Stage 2 - Reranking: Top {len(reranked_docs)} docs in {timing['reranking']:.2f}s")
            if reranked_docs and reranked_docs[0] != retrieved_docs[0]:
                logger.info(f"   New top doc: [{reranked_docs[0][2]}] {reranked_docs[0][1][:50]}...")

        # STAGE 3: KNOWLEDGE SELECTION
        start_time = time.time()

        if use_generative_selection:
            # Prepare hyperlinked history if available
            hyperlinked_history = None
            if dialogue_history and self.hyperlink_processor:
                hyperlinked_history = self.hyperlink_processor.get_hyperlinks_for_context(dialogue_history)

            selected_knowledge = self.select_knowledge_generatively(
                query, reranked_docs, dialogue_history, hyperlinked_history
            )
            selection_method = "generative"
        else:
            selected_knowledge = self.select_knowledge_by_similarity(
                query, reranked_docs, dialogue_history
            )
            selection_method = "similarity"

        timing['selection'] = time.time() - start_time

        if verbose:
            logger.info(
                f"✅ Stage 3 - Knowledge Selection ({selection_method}): {len(selected_knowledge)} pieces in {timing['selection']:.2f}s")
            for i, (title, text, score) in enumerate(selected_knowledge):
                logger.info(f"   K{i + 1}: [{title}] {text[:60]}... (score: {score:.3f})")

        # STAGE 4: RESPONSE GENERATION
        start_time = time.time()
        input_text = self.prepare_generation_input(query, selected_knowledge, dialogue_history)
        response = self.generate_response(input_text)
        timing['generation'] = time.time() - start_time

        if verbose:
            logger.info(f"✅ Stage 4 - Generation: Response in {timing['generation']:.2f}s")
            logger.info(f"   Response: {response[:100]}...")

        # Calculate total time
        timing['total'] = sum(timing.values())

        if verbose:
            logger.info(f"⏱️ Total pipeline time: {timing['total']:.2f}s")
            logger.info("=" * 60)

        # Update hyperlink processor for next turn
        if selected_knowledge and dialogue_history is not None:
            current_turn_id = len(dialogue_history)
            self.hyperlink_processor.add_hyperlink(
                current_turn_id,
                "k1",
                selected_knowledge[0][0]
            )

        return {
            "response": response,
            "retrieved_docs": retrieved_docs[:5],  # Top 5 for analysis
            "reranked_docs": reranked_docs[:3],  # Top 3 for analysis
            "selected_knowledge": selected_knowledge,
            "timing": timing,
            "method": {
                "retrieval": self.retrieval_method,
                "selection": selection_method
            }
        }

    def save(self, path: str):
        """Save model và configuration"""
        os.makedirs(path, exist_ok=True)

        # Save model và tokenizer
        self.model.save_pretrained(os.path.join(path, "model"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))

        # Save configuration
        config = {
            "top_k_retrieval": self.top_k_retrieval,
            "top_k_rerank": self.top_k_rerank,
            "top_k_knowledge": self.top_k_knowledge,
            "retrieval_method": self.retrieval_method,
            "use_generative_selection": self.use_generative_selection
        }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {path}")

    def load(self, path: str, device: Optional[str] = None):
        """Load model và configuration"""
        if device:
            self.device = device

        # Load model và tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            os.path.join(path, "model")
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(path, "tokenizer")
        )

        # Load configuration
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)

        self.top_k_retrieval = config.get("top_k_retrieval", self.top_k_retrieval)
        self.top_k_rerank = config.get("top_k_rerank", self.top_k_rerank)
        self.top_k_knowledge = config.get("top_k_knowledge", self.top_k_knowledge)
        self.retrieval_method = config.get("retrieval_method", self.retrieval_method)
        self.use_generative_selection = config.get("use_generative_selection", self.use_generative_selection)

        # Re-initialize selector
        self.generative_selector = GenerativeKnowledgeSelector(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

        logger.info(f"Model loaded from {path}")


# ================================================================================
# PHẦN 5: END-TO-END EVALUATION SYSTEM
# ================================================================================

class StageMetrics:
    """
    Lưu trữ và tính toán metrics cho từng stage của pipeline.
    Giúp phân tích chi tiết performance của từng component.
    """

    def __init__(self):
        self.retrieval_metrics = defaultdict(list)
        self.reranking_metrics = defaultdict(list)
        self.selection_metrics = defaultdict(list)
        self.generation_metrics = defaultdict(list)
        self.end_to_end_metrics = defaultdict(list)

    def add_metrics(self, stage: str, metrics: Dict):
        """Add metrics cho một stage"""
        stage_dict = getattr(self, f"{stage}_metrics")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                stage_dict[key].append(value)

    def compute_averages(self) -> Dict:
        """Tính average của tất cả metrics"""
        results = {}

        for stage in ['retrieval', 'reranking', 'selection', 'generation', 'end_to_end']:
            stage_metrics = getattr(self, f"{stage}_metrics")
            stage_results = {}

            for metric_name, values in stage_metrics.items():
                if values:
                    stage_results[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }

            if stage_results:
                results[stage] = stage_results

        return results


class EndToEndEvaluator:
    """
    Evaluator cho end-to-end evaluation của RA-GenKS.
    Đánh giá từng stage độc lập và cả pipeline.
    """

    def __init__(self, model: EnhancedRAGenKS):
        self.model = model
        self.metrics_tracker = StageMetrics()

    def evaluate_retrieval_stage(
            self,
            query: str,
            gold_knowledge: str,
            gold_title: str,
            dialogue_history: Optional[List[str]] = None,
            top_k_values: List[int] = [1, 5, 10, 20, 50, 100]
    ) -> Tuple[Dict, List]:
        """
        Đánh giá stage 1: Retrieval

        Metrics:
        - Recall@k: Gold document có trong top-k không
        - MRR: Mean Reciprocal Rank của gold document
        - Coverage: Bao nhiêu % gold knowledge được cover
        """
        # Retrieve documents
        retrieved_docs = self.model.retrieve_documents(query, dialogue_history, top_k=max(top_k_values))

        metrics = {}
        gold_found = False
        gold_rank = -1

        # Find gold document position
        for idx, (doc_id, doc_text, doc_title, doc_score) in enumerate(retrieved_docs):
            # Check if this is the gold knowledge
            if gold_knowledge in doc_text or (gold_title and gold_title in doc_title):
                if not gold_found:
                    gold_found = True
                    gold_rank = idx + 1
                    break

        # Calculate Recall@k
        for k in top_k_values:
            recall_at_k = 1.0 if gold_found and gold_rank <= k else 0.0
            metrics[f'recall@{k}'] = recall_at_k

        # Calculate MRR
        mrr = 1.0 / gold_rank if gold_rank > 0 else 0.0
        metrics['mrr'] = mrr

        # Calculate coverage
        if retrieved_docs:
            all_text = " ".join([doc[1] for doc in retrieved_docs[:20]])
            gold_tokens = set(normalize_answer(gold_knowledge).split())
            retrieved_tokens = set(normalize_answer(all_text).split())

            if gold_tokens:
                coverage = len(gold_tokens & retrieved_tokens) / len(gold_tokens)
            else:
                coverage = 0.0
            metrics['coverage'] = coverage
        else:
            metrics['coverage'] = 0.0

        metrics['retrieval_success'] = 1.0 if gold_found else 0.0

        return metrics, retrieved_docs

    def evaluate_reranking_stage(
            self,
            query: str,
            retrieved_docs: List,
            gold_knowledge: str,
            gold_title: str,
            dialogue_history: Optional[List[str]] = None
    ) -> Tuple[Dict, List]:
        """
        Đánh giá stage 2: Reranking

        Metrics:
        - Position improvement: Gold document lên bao nhiêu vị trí
        - NDCG: Normalized Discounted Cumulative Gain
        - Reranked MRR
        """
        # Find initial position
        initial_gold_rank = -1
        for idx, (_, doc_text, doc_title, _) in enumerate(retrieved_docs):
            if gold_knowledge in doc_text or (gold_title and gold_title in doc_title):
                initial_gold_rank = idx + 1
                break

        # Rerank documents
        reranked_docs = self.model.rerank_documents(query, retrieved_docs, dialogue_history)

        # Find new position
        reranked_gold_rank = -1
        for idx, (_, doc_text, doc_title, _) in enumerate(reranked_docs):
            if gold_knowledge in doc_text or (gold_title and gold_title in doc_title):
                reranked_gold_rank = idx + 1
                break

        metrics = {}

        # Position improvement
        if initial_gold_rank > 0 and reranked_gold_rank > 0:
            metrics['position_improvement'] = initial_gold_rank - reranked_gold_rank
            metrics['rerank_success'] = 1.0 if reranked_gold_rank <= 5 else 0.0
        else:
            metrics['position_improvement'] = 0
            metrics['rerank_success'] = 0.0

        # Reranked MRR
        metrics['reranked_mrr'] = 1.0 / reranked_gold_rank if reranked_gold_rank > 0 else 0.0

        # NDCG calculation
        if reranked_docs:
            # Create relevance scores
            relevance_scores = []
            for doc in reranked_docs[:10]:
                is_relevant = gold_knowledge in doc[1] or (gold_title and gold_title in doc[2])
                relevance_scores.append(1.0 if is_relevant else 0.0)

            if sum(relevance_scores) > 0:
                ideal_scores = sorted(relevance_scores, reverse=True)
                try:
                    metrics['ndcg@10'] = ndcg_score([ideal_scores], [relevance_scores])
                except:
                    metrics['ndcg@10'] = 0.0
            else:
                metrics['ndcg@10'] = 0.0

        return metrics, reranked_docs

    def evaluate_selection_stage(
            self,
            query: str,
            reranked_docs: List,
            gold_knowledge: str,
            dialogue_history: Optional[List[str]] = None,
            use_generative: bool = True
    ) -> Tuple[Dict, List]:
        """
        Đánh giá stage 3: Knowledge Selection

        Metrics:
        - Exact match: Chọn chính xác gold knowledge
        - Partial match: Có overlap với gold knowledge
        - Selection F1
        """
        # Select knowledge
        if use_generative:
            selected_knowledge = self.model.select_knowledge_generatively(
                query, reranked_docs, dialogue_history
            )
        else:
            selected_knowledge = self.model.select_knowledge_by_similarity(
                query, reranked_docs, dialogue_history
            )

        metrics = {}

        # Exact match
        exact_match = any(
            gold_knowledge == knowledge[1]
            for knowledge in selected_knowledge
        )
        metrics['exact_match'] = 1.0 if exact_match else 0.0

        # Partial match và overlap
        max_overlap = 0.0
        gold_tokens = set(normalize_answer(gold_knowledge).split())

        for title, text, score in selected_knowledge:
            selected_tokens = set(normalize_answer(text).split())

            if gold_tokens:
                overlap = len(gold_tokens & selected_tokens) / len(gold_tokens)
                max_overlap = max(max_overlap, overlap)

        metrics['partial_match'] = 1.0 if max_overlap >= 0.5 else 0.0
        metrics['max_overlap'] = max_overlap

        # Selection F1
        if selected_knowledge and gold_tokens:
            all_selected = " ".join([text for _, text, _ in selected_knowledge])
            selected_tokens = set(normalize_answer(all_selected).split())

            common = gold_tokens & selected_tokens
            if common:
                precision = len(common) / len(selected_tokens)
                recall = len(common) / len(gold_tokens)
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            metrics['selection_precision'] = precision if 'precision' in locals() else 0.0
            metrics['selection_recall'] = recall if 'recall' in locals() else 0.0
            metrics['selection_f1'] = f1
        else:
            metrics['selection_precision'] = 0.0
            metrics['selection_recall'] = 0.0
            metrics['selection_f1'] = 0.0

        return metrics, selected_knowledge

    def evaluate_generation_stage(
            self,
            query: str,
            selected_knowledge: List,
            target_response: str,
            gold_knowledge: str,
            dialogue_history: Optional[List[str]] = None
    ) -> Tuple[Dict, str]:
        """
        Đánh giá stage 4: Response Generation

        Metrics:
        - BLEU, ROUGE, F1
        - Knowledge F1 (hallucination detection)
        - Faithfulness
        """
        # Generate response
        input_text = self.model.prepare_generation_input(query, selected_knowledge, dialogue_history)
        generated_response = self.model.generate_response(input_text)

        metrics = {}

        # BLEU scores
        ref_tokens = target_response.lower().split()
        hyp_tokens = generated_response.lower().split()

        smoothie = SmoothingFunction().method1
        metrics['bleu1'] = sentence_bleu([ref_tokens], hyp_tokens,
                                         weights=(1, 0, 0, 0),
                                         smoothing_function=smoothie)
        metrics['bleu4'] = sentence_bleu([ref_tokens], hyp_tokens,
                                         weights=(0.25, 0.25, 0.25, 0.25),
                                         smoothing_function=smoothie)

        # ROUGE scores
        try:
            rouge = Rouge()
            rouge_scores = rouge.get_scores(generated_response.lower(), target_response.lower())[0]
            metrics['rouge1'] = rouge_scores['rouge-1']['f']
            metrics['rouge2'] = rouge_scores['rouge-2']['f']
            metrics['rougeL'] = rouge_scores['rouge-l']['f']
        except:
            metrics['rouge1'] = 0.0
            metrics['rouge2'] = 0.0
            metrics['rougeL'] = 0.0

        # F1 scores
        metrics['unigram_f1'] = calculate_unigram_f1(generated_response, target_response)
        metrics['knowledge_f1'] = calculate_knowledge_f1(generated_response, gold_knowledge)

        # Faithfulness và hallucination
        if selected_knowledge:
            selected_text = " ".join([text for _, text, _ in selected_knowledge])
            response_tokens = set(normalize_answer(generated_response).split())
            knowledge_tokens = set(normalize_answer(selected_text).split())

            if response_tokens:
                # Faithfulness: % response traceable to knowledge
                traceable = response_tokens & knowledge_tokens
                metrics['faithfulness'] = len(traceable) / len(response_tokens)

                # Hallucination: tokens not in knowledge or dialogue
                all_context = selected_text
                if dialogue_history:
                    all_context += " " + " ".join(dialogue_history)

                context_tokens = set(normalize_answer(all_context).split())
                hallucinated = response_tokens - context_tokens
                metrics['hallucination_rate'] = len(hallucinated) / len(response_tokens)
            else:
                metrics['faithfulness'] = 0.0
                metrics['hallucination_rate'] = 1.0
        else:
            metrics['faithfulness'] = 0.0
            metrics['hallucination_rate'] = 1.0

        # Perplexity
        ppl = self.model.calculate_perplexity(input_text, target_response)
        metrics['perplexity'] = ppl

        return metrics, generated_response

    def evaluate_sample(
            self,
            example: Dict,
            use_generative_selection: bool = True,
            verbose: bool = False
    ) -> Dict:
        """
        Đánh giá một sample qua toàn bộ pipeline.

        Returns:
            Dict chứa metrics cho từng stage và overall
        """
        # Extract information từ example
        query = example.get('query', '') or example.get('text', '')
        dialogue_history = []

        if 'context' in example:
            for turn in example['context']:
                dialogue_history.append(turn.get('text', ''))

        gold_knowledge = example.get('checked_sentence', '')
        gold_title = example.get('title', '')
        target_response = example.get('response', '') or (
            example.get('labels', [''])[0] if 'labels' in example else ''
        )

        # Skip nếu không có gold knowledge
        if not gold_knowledge or gold_knowledge == 'no_passages_used':
            return {}

        results = {}

        # Stage 1: Retrieval
        retrieval_metrics, retrieved_docs = self.evaluate_retrieval_stage(
            query, gold_knowledge, gold_title, dialogue_history
        )
        results['retrieval'] = retrieval_metrics

        # Stage 2: Reranking
        reranking_metrics, reranked_docs = self.evaluate_reranking_stage(
            query, retrieved_docs[:100], gold_knowledge, gold_title, dialogue_history
        )
        results['reranking'] = reranking_metrics

        # Stage 3: Selection
        selection_metrics, selected_knowledge = self.evaluate_selection_stage(
            query, reranked_docs, gold_knowledge, dialogue_history, use_generative_selection
        )
        results['selection'] = selection_metrics

        # Stage 4: Generation
        generation_metrics, generated_response = self.evaluate_generation_stage(
            query, selected_knowledge, target_response, gold_knowledge, dialogue_history
        )
        results['generation'] = generation_metrics

        # Overall metrics
        results['end_to_end'] = {
            'pipeline_success': (
                    results['retrieval']['retrieval_success'] *
                    results['selection']['partial_match'] *
                    (1 - results['generation']['hallucination_rate'])
            ),
            'combined_score': (
                    0.2 * results['retrieval'].get('recall@10', 0) +
                    0.2 * results['selection'].get('selection_f1', 0) +
                    0.3 * results['generation'].get('unigram_f1', 0) +
                    0.3 * results['generation'].get('knowledge_f1', 0)
            )
        }

        # Store in tracker
        for stage, metrics in results.items():
            self.metrics_tracker.add_metrics(stage, metrics)

        if verbose:
            logger.info(f"Sample evaluation complete. Combined score: {results['end_to_end']['combined_score']:.3f}")

        return results


def run_end_to_end_evaluation(
        model: EnhancedRAGenKS,
        eval_data: List[Dict],
        output_dir: str = './evaluation_results',
        num_samples: Optional[int] = None,
        run_ablation: bool = True,
        verbose: bool = False
) -> Dict:
    """
    Chạy end-to-end evaluation trên dataset.

    Args:
        model: Trained RA-GenKS model
        eval_data: Evaluation dataset
        output_dir: Directory để save results
        num_samples: Number of samples to evaluate (None = all)
        run_ablation: Run ablation study với different configurations
        verbose: Print detailed logs

    Returns:
        Dict chứa aggregated results
    """
    logger.info("=" * 80)
    logger.info("RUNNING END-TO-END EVALUATION")
    logger.info("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    if num_samples:
        eval_data = eval_data[:num_samples]

    logger.info(f"Evaluating {len(eval_data)} samples...")

    # Configurations to test
    if run_ablation:
        configs = [
            {'name': 'full_pipeline', 'use_generative': True, 'use_rerank': True},
            {'name': 'no_generative', 'use_generative': False, 'use_rerank': True},
            {'name': 'no_rerank', 'use_generative': True, 'use_rerank': False},
            {'name': 'baseline', 'use_generative': False, 'use_rerank': False}
        ]
    else:
        configs = [{'name': 'full_pipeline', 'use_generative': True, 'use_rerank': True}]

    all_results = {}

    for config in configs:
        logger.info(f"\nEvaluating configuration: {config['name']}")

        # Create evaluator
        evaluator = EndToEndEvaluator(model)

        # Temporarily modify model settings for ablation
        original_use_generative = model.use_generative_selection
        model.use_generative_selection = config['use_generative']

        # Evaluate samples
        sample_results = []
        for idx, example in enumerate(tqdm(eval_data, desc=f"Evaluating {config['name']}")):
            result = evaluator.evaluate_sample(example, config['use_generative'], verbose=verbose)
            if result:  # Skip samples without gold knowledge
                sample_results.append(result)

        # Restore original settings
        model.use_generative_selection = original_use_generative

        # Aggregate results
        aggregated = evaluator.metrics_tracker.compute_averages()
        all_results[config['name']] = {
            'aggregated': aggregated,
            'num_samples': len(sample_results)
        }

        # Print summary
        logger.info(f"\n📊 Results for {config['name']}:")
        logger.info("-" * 60)

        for stage in ['retrieval', 'reranking', 'selection', 'generation', 'end_to_end']:
            if stage in aggregated:
                logger.info(f"\n{stage.upper()} Stage:")
                stage_metrics = aggregated[stage]

                # Select key metrics to display
                key_metrics = {
                    'retrieval': ['recall@10', 'mrr', 'coverage'],
                    'reranking': ['position_improvement', 'reranked_mrr', 'ndcg@10'],
                    'selection': ['exact_match', 'partial_match', 'selection_f1'],
                    'generation': ['unigram_f1', 'knowledge_f1', 'faithfulness', 'hallucination_rate'],
                    'end_to_end': ['pipeline_success', 'combined_score']
                }

                for metric in key_metrics.get(stage, []):
                    if metric in stage_metrics:
                        value = stage_metrics[metric]['mean']
                        std = stage_metrics[metric]['std']
                        logger.info(f"  {metric:25s}: {value:.3f} (±{std:.3f})")

    # Ablation analysis
    if run_ablation and len(configs) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("ABLATION STUDY RESULTS")
        logger.info("=" * 80)

        baseline = all_results.get('baseline', {}).get('aggregated', {})
        full = all_results.get('full_pipeline', {}).get('aggregated', {})

        if baseline and full:
            # Calculate improvements
            improvements = {}

            # Overall improvement
            baseline_score = baseline.get('end_to_end', {}).get('combined_score', {}).get('mean', 0)
            full_score = full.get('end_to_end', {}).get('combined_score', {}).get('mean', 0)
            overall_improvement = (full_score - baseline_score) * 100

            logger.info(f"\n📈 Overall Improvement: +{overall_improvement:.2f}%")
            logger.info(f"  Baseline score: {baseline_score:.3f}")
            logger.info(f"  Full pipeline: {full_score:.3f}")

            # Component contributions
            if 'no_generative' in all_results:
                no_gen = all_results['no_generative']['aggregated']
                no_gen_score = no_gen.get('end_to_end', {}).get('combined_score', {}).get('mean', 0)
                gen_contribution = (full_score - no_gen_score) * 100
                logger.info(f"\n  Generative selection contribution: +{gen_contribution:.2f}%")

            if 'no_rerank' in all_results:
                no_rerank = all_results['no_rerank']['aggregated']
                no_rerank_score = no_rerank.get('end_to_end', {}).get('combined_score', {}).get('mean', 0)
                rerank_contribution = (full_score - no_rerank_score) * 100
                logger.info(f"  Reranking contribution: +{rerank_contribution:.2f}%")

    # Save results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n✅ Results saved to {results_file}")

    return all_results


# ================================================================================
# PHẦN 6: TRAINING FUNCTIONS
# ================================================================================

def train_ragenks(
        model: EnhancedRAGenKS,
        train_data: List[Dict],
        valid_data: List[Dict],
        output_dir: str = './checkpoints',
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        evaluate_every_n_steps: int = 500,
        save_every_n_steps: int = 1000,
        gpu_monitor: Optional[GPUMonitor] = None
) -> EnhancedRAGenKS:
    """
    Train RA-GenKS model với teacher forcing.

    Trong training, model học từ gold knowledge (không run full pipeline).
    Evaluation sẽ test full pipeline riêng.
    """
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

    logger.info(f"Training RA-GenKS for {epochs} epochs with {len(train_data)} samples")

    if gpu_monitor:
        gpu_monitor.log_stats("training_start")

    # Create datasets
    train_dataset = RAGenKSDataset(
        train_data,
        model.tokenizer,
        test=False,
        top_k_knowledge=model.top_k_knowledge
    )

    valid_dataset = RAGenKSDataset(
        valid_data,
        model.tokenizer,
        test=True,
        top_k_knowledge=model.top_k_knowledge
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Setup optimizer và scheduler
    total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps

    optimizer = AdamW(
        model.model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Prepare for distributed training
    model.model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
        model.model, optimizer, train_dataloader, valid_dataloader, scheduler
    )

    # Training variables
    global_step = 0
    best_valid_loss = float('inf')
    training_history = []

    logger.info("Starting training...")

    for epoch in range(epochs):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'=' * 60}")

        # Training phase
        model.model.train()
        train_loss = 0
        train_steps = 0

        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            with accelerator.accumulate(model.model):
                outputs = model.model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            train_steps += 1
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': train_loss / train_steps,
                'lr': scheduler.get_last_lr()[0]
            })

            # Periodic evaluation
            if global_step % evaluate_every_n_steps == 0:
                logger.info(f"\nEvaluating at step {global_step}...")

                model.model.eval()
                valid_loss = 0
                valid_steps = 0

                with torch.no_grad():
                    for batch in tqdm(valid_dataloader, desc="Validation"):
                        outputs = model.model(**batch)
                        valid_loss += outputs.loss.item()
                        valid_steps += 1

                avg_valid_loss = valid_loss / valid_steps

                logger.info(f"Step {global_step} - Valid loss: {avg_valid_loss:.4f}")

                # Save best model
                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss

                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, 'best_model')
                        os.makedirs(save_path, exist_ok=True)

                        unwrapped_model = accelerator.unwrap_model(model.model)
                        model.model = unwrapped_model  # Temporarily unwrap
                        model.save(save_path)
                        model.model = accelerator.prepare(unwrapped_model)  # Re-wrap

                        logger.info(f"✅ Saved new best model (loss: {best_valid_loss:.4f})")

                model.model.train()

            # Periodic checkpoint
            # if global_step % save_every_n_steps == 0 and accelerator.is_main_process:
            #     save_path = os.path.join(output_dir, f'checkpoint_{global_step}')
            #     os.makedirs(save_path, exist_ok=True)
            #
            #     unwrapped_model = accelerator.unwrap_model(model.model)
            #     model.model = unwrapped_model
            #     model.save(save_path)
            #     model.model = accelerator.prepare(unwrapped_model)
            #
            #     logger.info(f"💾 Saved checkpoint at step {global_step}")

        # End of epoch evaluation
        avg_train_loss = train_loss / train_steps
        logger.info(f"\nEpoch {epoch + 1} complete. Average training loss: {avg_train_loss:.4f}")

        if gpu_monitor:
            gpu_monitor.log_stats(f"epoch_{epoch + 1}_complete", {
                'train_loss': avg_train_loss,
                'best_valid_loss': best_valid_loss
            })

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'best_valid_loss': best_valid_loss
        })

    # Save final model
    if accelerator.is_main_process:
        save_path = os.path.join(output_dir, 'final_model')
        os.makedirs(save_path, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model.model)
        model.model = unwrapped_model
        model.save(save_path)

        # Save training history
        history_file = os.path.join(output_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)

    logger.info("✅ Training complete!")

    return model

def main():
    """
    Main function để chạy complete RA-GenKS system.

    Pipeline:
    1. Load data
    2. Build corpus index
    3. Train model
    4. Run end-to-end evaluation
    5. Demo với process_query
    """
    logger.info("=" * 80)
    logger.info("RA-GENKS COMPLETE SYSTEM")
    logger.info("Retrieval-Augmented Generation with Knowledge Selection")
    logger.info("=" * 80)

    # Configuration
    config = {
        'model_name': 'facebook/bart-base',
        'retrieval_method': 'bm25',  # 'bm25', 'dpr', 'sentence_bert'
        'use_generative_selection': True,
        'top_k_retrieval': 100,
        'top_k_rerank': 20,
        'top_k_knowledge': 3,
        'batch_size': 4,
        'epochs': 3,
        'learning_rate': 2e-5,
        'data_dir': '/kaggle/input/wizard',
        'output_dir': '/kaggle/working/ckpt/enhanced_genks',
        'cache_dir': '/kaggle/working/ckpt/cache'
    }

    # Create directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['cache_dir'], exist_ok=True)

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor()
    gpu_monitor.start_monitoring()

    # Step 1: Load data
    logger.info("\n📚 Step 1: Loading data...")
    train_data = json.load(open('/kaggle/input/wizard/train.json'))
    valid_data = json.load(open('/kaggle/input/wizard/valid_seen.json'))
    test_data = json.load(open('/kaggle/input/wizard/test_seen.json'))

    # Use subset for quick testing (remove this for full training)
    # if len(train_data) > 1000:
    #     logger.info("Using subset of data for quick testing...")
    #     train_data = train_data[:1000]
    #     valid_data = valid_data[:200]
    #     test_data = test_data[:200]

    # Step 2: Initialize model
    logger.info("\n🤖 Step 2: Initializing RA-GenKS model...")
    model = EnhancedRAGenKS(
        model_name=config['model_name'],
        retrieval_method=config['retrieval_method'],
        use_generative_selection=config['use_generative_selection'],
        top_k_retrieval=config['top_k_retrieval'],
        top_k_rerank=config['top_k_rerank'],
        top_k_knowledge=config['top_k_knowledge'],
        cache_dir=config['cache_dir']
    )

    # Step 3: Build corpus index
    logger.info("\n🔍 Step 3: Building corpus index...")
    corpus_cache = os.path.join(config['cache_dir'], f'corpus_embeddings_{config["retrieval_method"]}.pt')
    model.build_corpus_index(train_data + valid_data, cache_path=corpus_cache)

    # Step 4: Train model
    logger.info("\n🎯 Step 4: Training model...")
    model = train_ragenks(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        output_dir=os.path.join(config['output_dir'], 'checkpoints'),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        gpu_monitor=gpu_monitor
    )

    # Step 5: End-to-end evaluation
    logger.info("\n📊 Step 5: Running end-to-end evaluation...")
    eval_results = run_end_to_end_evaluation(
        model=model,
        eval_data=test_data,
        output_dir=os.path.join(config['output_dir'], 'evaluation'),
        num_samples=None,
        run_ablation=False,
        verbose=False
    )

    # Step 6: Demo with process_query
    logger.info("\n🎭 Step 6: Demo - Processing sample queries...")

    # # Demo 1: Simple query
    # demo_result = model.process_query(
    #     query="Tell me about machine learning",
    #     dialogue_history=None,
    #     verbose=True
    # )
    #
    # # Demo 2: Query with context
    # demo_result_2 = model.process_query(
    #     query="What are the main types?",
    #     dialogue_history=["Let's talk about artificial intelligence", "Sure! AI is a broad field."],
    #     verbose=True
    # )

    # Step 7: Final summary
    logger.info("\n" + "=" * 80)
    logger.info("EXECUTION COMPLETE")
    logger.info("=" * 80)

    # Print summary statistics
    if 'full_pipeline' in eval_results:
        results = eval_results['full_pipeline']['aggregated']

        logger.info("\n📈 Final Performance Summary:")
        logger.info("-" * 40)

        key_metrics = [
            ('Retrieval Recall@10', results.get('retrieval', {}).get('recall@10', {}).get('mean', 0)),
            ('Selection F1', results.get('selection', {}).get('selection_f1', {}).get('mean', 0)),
            ('Generation F1', results.get('generation', {}).get('unigram_f1', {}).get('mean', 0)),
            ('Knowledge F1', results.get('generation', {}).get('knowledge_f1', {}).get('mean', 0)),
            ('Hallucination Rate', results.get('generation', {}).get('hallucination_rate', {}).get('mean', 0)),
            ('Combined Score', results.get('end_to_end', {}).get('combined_score', {}).get('mean', 0))
        ]

        for metric_name, value in key_metrics:
            if 'Rate' in metric_name:
                logger.info(f"{metric_name:20s}: {value:.3f}")
            else:
                logger.info(f"{metric_name:20s}: {value * 100:.2f}%")

    # GPU statistics
    if gpu_monitor.gpu_available:
        logger.info(f"\n💻 GPU Statistics:")
        logger.info(f"Peak memory usage: {gpu_monitor.peak_memory:.2f} GB")

    logger.info(f"\n✅ All results saved to: {config['output_dir']}")
    logger.info("🎉 RA-GenKS system execution complete!")


if __name__ == '__main__':
    # Set random seeds for reproducibility
    import random

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Run main pipeline
    main()