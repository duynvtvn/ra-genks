import logging
import os
import json
import re
import torch
import numpy as np
import faiss
from tqdm import tqdm
from collections import OrderedDict, defaultdict, Counter
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
import time
from datetime import datetime, timedelta
import psutil

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

class GPUMonitor:
    """Lớp theo dõi và ghi lại thông tin sử dụng GPU trong quá trình huấn luyện"""

    def __init__(self, device='cuda'):
        self.device = device
        self.monitoring_data = []
        self.peak_memory = 0
        self.start_time = None

        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            self.gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU Monitor khởi tạo: {self.gpu_count} GPU(s) - {self.gpu_name}")
        else:
            logger.warning("Không tìm thấy GPU, chỉ theo dõi CPU/RAM")

    def get_gpu_stats(self):
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_used_gb': psutil.virtual_memory().used / (1024 ** 3),
            'ram_percent': psutil.virtual_memory().percent
        }

        if self.gpu_available:
            stats['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            stats['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            stats['gpu_memory_cached_gb'] = torch.cuda.memory_reserved(self.device) / (1024 ** 3)

            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            stats['gpu_memory_total_gb'] = total_memory
            stats['gpu_memory_free_gb'] = total_memory - stats['gpu_memory_allocated_gb']
            stats['gpu_memory_percent'] = (stats['gpu_memory_allocated_gb'] / total_memory) * 100

            if stats['gpu_memory_allocated_gb'] > self.peak_memory:
                self.peak_memory = stats['gpu_memory_allocated_gb']
            stats['peak_memory_gb'] = self.peak_memory

            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    stats['gpu_utilization_percent'] = gpu.load * 100
                    stats['gpu_temperature_c'] = gpu.temperature
            except:
                pass

        return stats

    def start_monitoring(self):
        self.start_time = time.time()
        initial_stats = self.get_gpu_stats()
        initial_stats['event'] = 'start_monitoring'
        self.monitoring_data.append(initial_stats)
        logger.info(f"Bắt đầu theo dõi GPU/System tại {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def log_stats(self, event_name="checkpoint", extra_info=None):
        stats = self.get_gpu_stats()
        stats['event'] = event_name

        if self.start_time:
            elapsed_time = time.time() - self.start_time
            stats['elapsed_time_seconds'] = elapsed_time
            stats['elapsed_time_formatted'] = str(timedelta(seconds=int(elapsed_time)))

        if extra_info:
            stats.update(extra_info)

        self.monitoring_data.append(stats)

        if self.gpu_available:
            logger.info(
                f"[{event_name}] GPU Memory: {stats['gpu_memory_allocated_gb']:.2f}/{stats['gpu_memory_total_gb']:.2f} GB "
                f"({stats['gpu_memory_percent']:.1f}%) | Peak: {self.peak_memory:.2f} GB")

    def save_monitoring_report(self, filepath):
        report = {
            'summary': {
                'total_time_seconds': time.time() - self.start_time if self.start_time else 0,
                'total_time_formatted': str(
                    timedelta(seconds=int(time.time() - self.start_time))) if self.start_time else "N/A",
                'gpu_name': self.gpu_name if self.gpu_available else "No GPU",
                'peak_gpu_memory_gb': self.peak_memory,
                'monitoring_start': self.monitoring_data[0]['timestamp'] if self.monitoring_data else None,
                'monitoring_end': self.monitoring_data[-1]['timestamp'] if self.monitoring_data else None,
            },
            'detailed_logs': self.monitoring_data
        }

        if self.monitoring_data and self.gpu_available:
            gpu_memory_values = [d['gpu_memory_allocated_gb'] for d in self.monitoring_data if
                                 'gpu_memory_allocated_gb' in d]
            gpu_percent_values = [d['gpu_memory_percent'] for d in self.monitoring_data if 'gpu_memory_percent' in d]

            report['summary']['avg_gpu_memory_gb'] = np.mean(gpu_memory_values) if gpu_memory_values else 0
            report['summary']['avg_gpu_memory_percent'] = np.mean(gpu_percent_values) if gpu_percent_values else 0

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Đã lưu báo cáo theo dõi GPU/System vào: {filepath}")


class HyperlinkProcessor:
    """Lớp xử lý và quản lý hyperlink giữa đối thoại và tri thức"""

    def __init__(self):
        self.dialogue_knowledge_mapping = {}

    def add_hyperlink(self, turn_id, knowledge_id, title):
        self.dialogue_knowledge_mapping[turn_id] = (knowledge_id, title)

    def get_hyperlinks_for_context(self, dialogue_history):
        hyperlinked_history = []

        for i, utterance in enumerate(dialogue_history):
            if i in self.dialogue_knowledge_mapping:
                knowledge_id, title = self.dialogue_knowledge_mapping[i]
                hyperlinked_utterance = f"[{title}]<{knowledge_id}> {utterance}"
                hyperlinked_history.append(hyperlinked_utterance)
            else:
                hyperlinked_history.append(utterance)

        return hyperlinked_history


class GenerativeKnowledgeSelector:
    """Bộ chọn tri thức dựa trên phương pháp tạo sinh (generative approach)"""

    def __init__(self, model, tokenizer, max_knowledge_candidates=20, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.max_knowledge_candidates = max_knowledge_candidates
        self.device = device

    def prepare_input_for_generation(self, query, dialogue_history, knowledge_candidates, hyperlinked_history=None):
        context_parts = []
        history_to_use = hyperlinked_history if hyperlinked_history else dialogue_history

        if history_to_use:
            for i, utterance in enumerate(history_to_use):
                speaker = "User1: " if i % 2 == 0 else "User2: "
                context_parts.append(f"{speaker}{utterance}")

        current_speaker = "User1: " if len(context_parts) % 2 == 0 else "User2: "
        context_parts.append(f"{current_speaker}{query}")
        context_text = "\n".join(context_parts)

        knowledge_parts = ["Reference Information:"]
        candidates_to_use = knowledge_candidates[:self.max_knowledge_candidates]

        for i, (title, text, _) in enumerate(candidates_to_use):
            knowledge_parts.append(f"<k{i + 1}> [{title}] {text}")

        knowledge_text = "\n".join(knowledge_parts)
        input_text = f"{context_text}\n\n{knowledge_text}\n\nKnowledge Selection:"

        return input_text

    def select_knowledge(self, query, knowledge_candidates, dialogue_history=None, hyperlinked_history=None, top_k=3):
        if not knowledge_candidates:
            return []

        input_text = self.prepare_input_for_generation(
            query, dialogue_history, knowledge_candidates, hyperlinked_history
        )

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
                max_length=20,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        logger.info(f"Generated identifier text: {generated_text}")

        selected_knowledge = []
        pattern = r'<k(\d+)>'
        matches = re.findall(pattern, generated_text)

        for match in matches:
            idx = int(match) - 1
            if idx < len(knowledge_candidates):
                title, text, score = knowledge_candidates[idx]
                selected_knowledge.append((title, text, score))
                if len(selected_knowledge) >= top_k:
                    break

        if len(selected_knowledge) < top_k:
            remaining_candidates = [
                c for c in knowledge_candidates
                if all(c[1] != k[1] for k in selected_knowledge)
            ]
            remaining_candidates.sort(key=lambda x: x[2], reverse=True)
            selected_knowledge.extend(remaining_candidates[:top_k - len(selected_knowledge)])

        return selected_knowledge


class ImprovedMultiSpanGENKSData(Dataset):
    """Lớp xử lý dữ liệu cho mô hình GenKS cải tiến"""

    def __init__(self, data, tokenizer, context_len=256, knowledge_len=64, max_length=1024,
                 test=False, top_k_knowledge=3, add_hyperlink=True):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.knowledge_len = knowledge_len
        self.max_length = max_length
        self.test = test
        self.top_k_knowledge = top_k_knowledge
        self.add_hyperlink = add_hyperlink
        self.hyperlink_processor = HyperlinkProcessor()

        special_tokens = ['<knowledge>', '</knowledge>']
        for i in range(1, top_k_knowledge + 1):
            special_tokens.extend([f'<k{i}>', f'</k{i}>'])
        self.tokenizer.add_tokens(special_tokens)

    def __getitem__(self, index):
        example = self.data[index]
        context_parts = []

        if 'chosen_topic' in example:
            context_parts.append(f"Topic: {example['chosen_topic']}")

        dialogue_history = []
        hyperlinked_history = []

        role = {'0_Wizard': 'User1', '1_Apprentice': 'User2', '0_Apprentice': 'User2',
                '1_Wizard': 'User1', 0: 'User1', 1: 'User2', 'user1': 'User1', 'user2': 'User2'}

        if 'context' in example:
            for i, turn in enumerate(example['context']):
                speaker = role.get(turn.get('speaker', ''), turn.get('speaker', ''))
                text = turn.get('text', '')
                dialogue_history.append(text)

                if self.add_hyperlink and 'knowledge_used' in turn:
                    knowledge_id = f"k{i + 1}"
                    knowledge_title = turn.get('knowledge_title', 'unknown')
                    self.hyperlink_processor.add_hyperlink(i, knowledge_id, knowledge_title)

            if self.add_hyperlink:
                hyperlinked_history = self.hyperlink_processor.get_hyperlinks_for_context(dialogue_history)

        history_to_use = hyperlinked_history if hyperlinked_history else dialogue_history

        for i, utterance in enumerate(history_to_use):
            speaker = role.get(i % 2, f"User{(i % 2) + 1}")
            context_parts.append(f"{speaker}: {utterance}")

        context_text = "\n".join(context_parts)

        knowledge_items = []

        if 'knowledge' in example:
            for title, sentences in example['knowledge'].items():
                for sentence in sentences:
                    knowledge_items.append((title, sentence, 1.0))

        checked_knowledge = []
        if 'title' in example and example.get('title') != 'no_passages_used' and 'checked_sentence' in example:
            checked_knowledge.append((example['title'], example['checked_sentence'], 1.0))
            knowledge_items = [k for k in knowledge_items if k[1] != example['checked_sentence']]

        selected_knowledge = checked_knowledge.copy()

        if len(selected_knowledge) < self.top_k_knowledge and knowledge_items:
            if not self.test:
                np.random.shuffle(knowledge_items)
            selected_knowledge.extend(knowledge_items[:self.top_k_knowledge - len(selected_knowledge)])

        input_sequence = context_text

        if selected_knowledge:
            knowledge_parts = ["<knowledge>Reference Information:"]
            for i, (title, sentence, _) in enumerate(selected_knowledge):
                if i < self.top_k_knowledge:
                    knowledge_parts.append(f"<k{i + 1}>[{title}] {sentence}</k{i + 1}>")
            knowledge_parts.append("</knowledge>")
            knowledge_text = "\n".join(knowledge_parts)
            input_sequence = f"{input_sequence}\n\n{knowledge_text}\n\nPhản hồi:"
        else:
            input_sequence = f"{input_sequence}\n\nPhản hồi:"

        if 'response' in example:
            target = example['response']
        elif 'labels' in example and example['labels']:
            target = example['labels'][0]
        else:
            target = ""

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
        return len(self.data)

    def collate_fn(self, data):
        from torch.nn.utils.rnn import pad_sequence
        padding_value = self.tokenizer.pad_token_id
        input_ids, labels = zip(*data)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(padding_value),
            'labels': labels,
        }


class EnhancedGenKSWithRAG:
    """Mô hình GenKS cải tiến kết hợp với RAG"""

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
        self.device = device
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.top_k_knowledge = top_k_knowledge
        self.retrieval_method = retrieval_method.lower()
        self.use_generative_selection = use_generative_selection

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        special_tokens = ['<knowledge>', '</knowledge>']
        for i in range(1, top_k_knowledge + 1):
            special_tokens.extend([f'<k{i}>', f'</k{i}>'])

        self.tokenizer.add_tokens(special_tokens)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self._initialize_retrievers(retriever_model_name, query_encoder_name, embed_dim)
        self._initialize_ranker(ranker_model_name)

        self.generative_selector = GenerativeKnowledgeSelector(
            model=self.model,
            tokenizer=self.tokenizer,
            max_knowledge_candidates=20,
            device=device
        )

        self.hyperlink_processor = HyperlinkProcessor()
        self.doc_cache = {}
        self.corpus = None
        self.id_to_doc = None

    def _initialize_retrievers(self, retriever_model_name, query_encoder_name, embed_dim):
        try:
            from rank_bm25 import BM25Okapi
            self.bm25_class = BM25Okapi
            self.bm25 = None
            self.tokenized_corpus = None
            logger.info("Đã khởi tạo BM25 thành công")
        except ImportError:
            logger.warning("Không thể nhập rank_bm25. Cài đặt bằng cách 'pip install rank-bm25'")
            self.bm25_class = None

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

        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    def _initialize_ranker(self, ranker_model_name):
        try:
            from sentence_transformers import CrossEncoder
            self.ranker = CrossEncoder(ranker_model_name)
            logger.info(f"Đã khởi tạo CrossEncoder thành công: {ranker_model_name}")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo CrossEncoder: {str(e)}, sẽ sử dụng phương pháp xếp hạng đơn giản")
            self.ranker = None

    def build_corpus_index(self, corpus, cache_path=None):
        self.corpus = corpus
        self.id_to_doc = {item[0]: (item[1], item[2]) for item in corpus}
        texts = [doc[1] for doc in corpus]

        if hasattr(self, 'bm25_class') and self.bm25_class is not None:
            logger.info("Đang xây dựng chỉ mục BM25...")
            self.tokenized_corpus = [text.lower().split() for text in texts]
            self.bm25 = self.bm25_class(self.tokenized_corpus)
            logger.info("Đã xây dựng chỉ mục BM25 thành công")

        if self.retrieval_method == 'dpr':
            if cache_path and os.path.exists(cache_path):
                logger.info(f"Đang tải embeddings từ cache: {cache_path}")
                self.corpus_embeddings = torch.load(cache_path)
                self.index = faiss.IndexFlatIP(self.corpus_embeddings.size(1))
                self.index.add(self.corpus_embeddings.cpu().numpy())
                logger.info("Đã tải và xây dựng chỉ mục từ cache thành công")
                return

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

                if cache_path:
                    logger.info(f"Đang lưu embeddings vào cache: {cache_path}")
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(self.corpus_embeddings, cache_path)

                self.index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
                self.index.add(self.corpus_embeddings.cpu().numpy())

            elif hasattr(self, 'encoder'):
                logger.info("Đang tạo embeddings cho corpus với SentenceTransformer...")
                self.corpus_embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_tensor=True)
                self.index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
                self.index.add(self.corpus_embeddings.cpu().numpy())

        logger.info("Đang tạo ma trận TF-IDF...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

    def retrieve_documents(self, query, dialogue_history=None, top_k=None):
        if top_k is None:
            top_k = self.top_k_retrieval

        if dialogue_history:
            last_n_turns = dialogue_history[-3:]
            combined_query = " ".join(last_n_turns) + " " + query
        else:
            combined_query = query

        cache_key = combined_query[:100]
        if cache_key in self.doc_cache:
            logger.info(f"Sử dụng kết quả từ bộ nhớ cache cho truy vấn: {cache_key}...")
            return self.doc_cache[cache_key]

        if (self.retrieval_method == 'bm25' or self.retrieval_method not in ['dpr', 'tfidf']) and hasattr(self,
                                                                                                          'bm25') and self.bm25 is not None:
            logger.info("Đang truy xuất với BM25...")
            tokenized_query = combined_query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            indices = np.argsort(-scores)[:top_k]

            retrieved_docs = []
            for idx in indices:
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(scores[idx])))

        elif self.retrieval_method == 'dpr' and hasattr(self, 'query_encoder'):
            logger.info("Đang truy xuất với DPR...")
            with torch.no_grad():
                inputs = self.query_tokenizer(combined_query, return_tensors="pt").to(self.device)
                query_embedding = self.query_encoder(**inputs).pooler_output.cpu().numpy()

            scores, indices = self.index.search(query_embedding, top_k)

            retrieved_docs = []
            for idx, score in zip(indices.flatten(), scores.flatten()):
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(score)))

        elif hasattr(self, 'encoder'):
            logger.info("Đang truy xuất với SentenceTransformer...")
            query_embedding = self.encoder.encode(combined_query, convert_to_tensor=True)
            scores, indices = self.index.search(query_embedding.cpu().numpy().reshape(1, -1), top_k)

            retrieved_docs = []
            for idx, score in zip(indices.flatten(), scores.flatten()):
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(score)))

        else:
            logger.info("Đang truy xuất với TF-IDF...")
            query_vec = self.tfidf_vectorizer.transform([combined_query])
            scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
            indices = np.argsort(-scores)[:top_k]

            retrieved_docs = []
            for idx in indices:
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(scores[idx])))

        self.doc_cache[cache_key] = retrieved_docs
        return retrieved_docs

    def rerank_documents(self, query, docs, dialogue_history=None, top_k=None):
        if top_k is None:
            top_k = self.top_k_rerank

        if dialogue_history:
            last_n_turns = dialogue_history[-3:]
            combined_query = " ".join(last_n_turns) + " " + query
        else:
            combined_query = query

        if self.ranker:
            logger.info("Đang xếp hạng lại tài liệu với CrossEncoder...")
            pairs = [(combined_query, doc[1]) for doc in docs]
            scores = self.ranker.predict(pairs)

            reranked_docs = [
                (docs[i][0], docs[i][1], docs[i][2], float(scores[i]))
                for i in range(len(docs))
            ]
            reranked_docs = sorted(reranked_docs, key=lambda x: x[3], reverse=True)[:top_k]
        else:
            logger.info("Không có CrossEncoder, sử dụng điểm truy xuất gốc để xếp hạng...")
            reranked_docs = sorted(docs, key=lambda x: x[3], reverse=True)[:top_k]

        return reranked_docs

    def extract_sentences_from_docs(self, docs):
        all_sentences = []

        for doc_id, doc_text, doc_title, doc_score in docs:
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                sentences = nltk.sent_tokenize(doc_text)
            except:
                sentences = [doc_text]

            for sentence in sentences:
                if len(sentence.split()) >= 3:
                    all_sentences.append((doc_title, sentence, doc_score))

        return all_sentences

    def select_knowledge_generatively(self, query, reranked_docs, dialogue_history=None, hyperlinked_history=None):
        knowledge_candidates = self.extract_sentences_from_docs(reranked_docs)

        if not knowledge_candidates:
            return []

        selected_knowledge = self.generative_selector.select_knowledge(
            query=query,
            knowledge_candidates=knowledge_candidates,
            dialogue_history=dialogue_history,
            hyperlinked_history=hyperlinked_history,
            top_k=self.top_k_knowledge
        )

        return selected_knowledge

    def select_knowledge_by_similarity(self, query, reranked_docs, dialogue_history=None):
        all_sentences = self.extract_sentences_from_docs(reranked_docs)

        if not all_sentences:
            return []

        if dialogue_history:
            last_n_turns = dialogue_history[-3:]
            combined_query = " ".join(last_n_turns) + " " + query
        else:
            combined_query = query

        if hasattr(self, 'encoder'):
            try:
                query_embedding = self.encoder.encode(combined_query, convert_to_tensor=True)
                sentence_texts = [sent[1] for sent in all_sentences]
                sentence_embeddings = self.encoder.encode(sentence_texts, convert_to_tensor=True)

                similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
                top_indices = similarities.argsort(descending=True)[:self.top_k_knowledge].tolist()
                selected_knowledge = [all_sentences[idx] for idx in top_indices]

                return selected_knowledge
            except Exception as e:
                logger.warning(f"Lỗi khi tính toán độ tương đồng: {str(e)}")

        sorted_sentences = sorted(all_sentences, key=lambda x: x[2], reverse=True)
        return sorted_sentences[:self.top_k_knowledge]

    def prepare_generation_input(self, query, selected_knowledge, dialogue_history=None):
        hyperlinked_history = None
        if dialogue_history and self.hyperlink_processor:
            hyperlinked_history = self.hyperlink_processor.get_hyperlinks_for_context(dialogue_history)

        context_parts = []
        history_to_use = hyperlinked_history if hyperlinked_history else dialogue_history

        if history_to_use:
            for i, utterance in enumerate(history_to_use):
                speaker = "User1: " if i % 2 == 0 else "User2: "
                context_parts.append(f"{speaker}{utterance}")

        current_speaker = "User1: " if len(context_parts) % 2 == 0 else "User2: "
        context_parts.append(f"{current_speaker}{query}")
        context_text = "\n".join(context_parts)

        knowledge_parts = ["<knowledge>Reference Information:"]

        for i, (title, text, _) in enumerate(selected_knowledge):
            if i < self.top_k_knowledge:
                knowledge_parts.append(f"<k{i + 1}>[{title}] {text}</k{i + 1}>")

        knowledge_parts.append("</knowledge>")
        knowledge_text = "\n".join(knowledge_parts)

        input_text = f"{context_text}\n\n{knowledge_text}\n\nPhản hồi:"

        return input_text

    def generate_response(self, input_text, max_length=128, num_beams=4, temperature=0.7):
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
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                temperature=temperature
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def calculate_perplexity(self, input_text, target_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        # Tokenize target_text và lấy input_ids
        labels = self.tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True  # Đảm bảo padding nếu cần
        ).input_ids.to(self.device)

        # Mask các padding tokens bằng -100 để ignore trong loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels
            )
            loss = outputs.loss

        # Tính PPL: exp(average loss per token)
        ppl = math.exp(loss.item()) if loss is not None else float('inf')

        return ppl

    def process_query(self, query, dialogue_history=None, use_generative_selection=None):
        if use_generative_selection is None:
            use_generative_selection = self.use_generative_selection

        retrieved_docs = self.retrieve_documents(query, dialogue_history)
        logger.info(f"Giai đoạn 1: Đã truy xuất {len(retrieved_docs)} tài liệu")

        reranked_docs = self.rerank_documents(query, retrieved_docs, dialogue_history)
        logger.info(f"Giai đoạn 2: Đã xếp hạng lại lấy {len(reranked_docs)} tài liệu tốt nhất")

        hyperlinked_history = None
        if dialogue_history and self.hyperlink_processor:
            hyperlinked_history = self.hyperlink_processor.get_hyperlinks_for_context(dialogue_history)

        if use_generative_selection:
            selected_knowledge = self.select_knowledge_generatively(
                query, reranked_docs, dialogue_history, hyperlinked_history
            )
            logger.info(f"Giai đoạn 3: Đã chọn {len(selected_knowledge)} tri thức bằng phương pháp tạo sinh")
        else:
            selected_knowledge = self.select_knowledge_by_similarity(query, reranked_docs, dialogue_history)
            logger.info(f"Giai đoạn 3: Đã chọn {len(selected_knowledge)} tri thức dựa trên độ tương đồng")

        for i, (title, text, score) in enumerate(selected_knowledge):
            logger.info(f"  Tri thức {i + 1}: [{title}] {text[:50]}... (score: {score:.4f})")

        input_text = self.prepare_generation_input(query, selected_knowledge, dialogue_history)
        response = self.generate_response(input_text)
        logger.info(f"Giai đoạn 4: Đã sinh phản hồi: {response[:50]}...")

        current_turn_id = len(dialogue_history) if dialogue_history else 0
        if selected_knowledge and len(selected_knowledge) > 0:
            self.hyperlink_processor.add_hyperlink(
                current_turn_id,
                f"k1",
                selected_knowledge[0][0]
            )

        ppl = None
        if 'target_response' in locals() and locals()['target_response']:
            target_response = locals()['target_response']
            ppl = self.calculate_perplexity(input_text, target_response)
            logger.info(f"PPL của phản hồi mục tiêu: {ppl:.2f}")

        return {
            "response": response,
            "retrieved_docs": retrieved_docs[:5],
            "reranked_docs": reranked_docs[:3],
            "selected_knowledge": selected_knowledge,
            "ppl": ppl,
            "hyperlinked_history": self.hyperlink_processor.get_hyperlinks_for_context(
                (dialogue_history or []) + [query]
            )
        }

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        self.model.save_pretrained(os.path.join(path, "model"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))

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
        if device:
            self.device = device

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            os.path.join(path, "model")
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(path, "tokenizer")
        )

        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)

        self.top_k_retrieval = config.get("top_k_retrieval", self.top_k_retrieval)
        self.top_k_rerank = config.get("top_k_rerank", self.top_k_rerank)
        self.top_k_knowledge = config.get("top_k_knowledge", self.top_k_knowledge)
        self.retrieval_method = config.get("retrieval_method", self.retrieval_method)
        self.use_generative_selection = config.get("use_generative_selection", self.use_generative_selection)

        self.generative_selector = GenerativeKnowledgeSelector(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

        logger.info(f"Đã tải mô hình từ {path}")


def calculate_unigram_f1(prediction, ground_truth):
    """
    Tính Unigram F1 score với normalize_answer

    Args:
        prediction: Văn bản dự đoán (string)
        ground_truth: Văn bản tham chiếu (string)

    Returns:
        f1_score: Điểm F1 (float)
    """
    # Normalize trước khi tokenize
    prediction_normalized = normalize_answer(prediction)
    ground_truth_normalized = normalize_answer(ground_truth)

    prediction_tokens = prediction_normalized.split()
    ground_truth_tokens = ground_truth_normalized.split()

    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0

    pred_counter = Counter(prediction_tokens)
    truth_counter = Counter(ground_truth_tokens)

    common_tokens = 0
    for token in pred_counter:
        if token in truth_counter:
            common_tokens += min(pred_counter[token], truth_counter[token])

    precision = common_tokens / sum(pred_counter.values()) if sum(pred_counter.values()) > 0 else 0
    recall = common_tokens / sum(truth_counter.values()) if sum(truth_counter.values()) > 0 else 0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def normalize_answer(s):
    """
    Chuẩn hóa text: loại bỏ articles, dấu câu và khoảng trắng thừa
    """

    def remove_articles(text):
        return RE_ART.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return RE_PUNC.sub(' ', text)  # Chuyển dấu câu thành khoảng trắng

    def lower(text):
        return text.lower()

    # Áp dụng theo thứ tự: lower -> remove_punc -> remove_articles -> white_space_fix
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_knowledge_f1(prediction, gold_knowledge):
    """
    Tính Knowledge F1 - đo overlap giữa response và gold knowledge

    Đây là metric quan trọng để đánh giá hallucination:
    - Nếu KF1 cao: model đang sử dụng đúng tri thức được cung cấp
    - Nếu KF1 thấp: model có thể đang hallucinate (tự tạo thông tin)

    Args:
        prediction: Response của model (string)
        gold_knowledge: Tri thức vàng mà human đã dùng (string)
                       Trong WoW dataset, đây là 'checked_sentence'

    Returns:
        f1_score: Điểm Knowledge F1 (float từ 0 đến 1)
    """
    # Nếu không có gold knowledge, trả về 0
    # (trường hợp "no_passages_used" trong WoW)
    if not gold_knowledge or gold_knowledge == "no_passages_used":
        return 0.0

    # Normalize cả prediction và gold knowledge theo cách của bài báo
    prediction_normalized = normalize_answer(prediction)
    gold_knowledge_normalized = normalize_answer(gold_knowledge)

    # Tokenize thành từ đơn
    prediction_tokens = prediction_normalized.split()
    knowledge_tokens = gold_knowledge_normalized.split()

    # Edge cases: cả hai đều rỗng hoặc một trong hai rỗng
    if len(prediction_tokens) == 0 and len(knowledge_tokens) == 0:
        return 1.0
    if len(prediction_tokens) == 0 or len(knowledge_tokens) == 0:
        return 0.0

    # Dùng Counter để đếm tần suất từ (quan trọng cho việc tính chính xác)
    pred_counter = Counter(prediction_tokens)
    knowledge_counter = Counter(knowledge_tokens)

    # Tính intersection với tần suất
    # Ví dụ: nếu "the" xuất hiện 2 lần trong prediction và 3 lần trong knowledge,
    # thì chỉ tính 2 lần match
    common = pred_counter & knowledge_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    # Precision: Bao nhiêu % từ trong response có xuất hiện trong gold knowledge
    precision = num_same / sum(pred_counter.values())

    # Recall: Bao nhiêu % từ trong gold knowledge được model nhắc đến
    recall = num_same / sum(knowledge_counter.values())

    # F1 score - harmonic mean của precision và recall
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def evaluate_enhanced_genks(
        model,
        eval_data,
        output_file=None,
        batch_size=8,
        calculate_ppl=True,
        gpu_monitor=None
):
    """
    Đánh giá mô hình với Knowledge F1 được tính CHÍNH XÁC

    Sự khác biệt chính: Hàm này truy xuất gold knowledge từ data
    và so sánh với response, thay vì so sánh với ground truth response
    """
    logger.info(f"Đang đánh giá mô hình với {len(eval_data)} mẫu")
    logger.info("Sử dụng Knowledge F1 chính xác - so sánh với gold knowledge")

    if gpu_monitor:
        gpu_monitor.log_stats("evaluation_start")

    # Tạo dataset với hyperlink và knowledge tracking
    eval_dataset = ImprovedMultiSpanGENKSData(
        eval_data,
        model.tokenizer,
        context_len=512,
        knowledge_len=256,
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
        num_workers=4
    )

    model.model.eval()

    # Lists để lưu kết quả
    output_texts = []
    target_texts = []
    gold_knowledges = []  # List mới để lưu gold knowledge
    ppls = []

    # Trước tiên, extract gold knowledge từ eval_data
    for example in eval_data:
        # Trong Wizard of Wikipedia dataset:
        # - 'checked_sentence' là câu tri thức được sử dụng
        # - 'title' là tiêu đề của passage chứa tri thức
        gold_knowledge = ""

        if 'checked_sentence' in example:
            gold_knowledge = example['checked_sentence']
            # Có thể kết hợp với title nếu muốn context đầy đủ hơn
            if 'title' in example and example['title'] != 'no_passages_used':
                # Option: có thể thêm title vào để có context
                # gold_knowledge = f"{example['title']}: {gold_knowledge}"
                pass
        elif 'knowledge_sentence' in example:
            # Một số dataset có thể dùng field khác
            gold_knowledge = example['knowledge_sentence']

        gold_knowledges.append(gold_knowledge)

    # Generate responses
    with torch.no_grad():
        sample_idx = 0
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader))):
            if gpu_monitor and batch_idx % 10 == 0:
                gpu_monitor.log_stats(f"eval_batch_{batch_idx}")

            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Generate responses
            outputs = model.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=512,
                num_beams=4,
                no_repeat_ngram_size=3,
                temperature=0.7
            )

            # Decode outputs và targets
            for i in range(outputs.size(0)):
                output_text = model.tokenizer.decode(outputs[i], skip_special_tokens=True)
                output_texts.append(output_text)

                # Decode target response
                target_ids = batch['labels'][i].clone()
                target_ids[target_ids == -100] = model.tokenizer.pad_token_id
                target_text = model.tokenizer.decode(target_ids, skip_special_tokens=True)
                target_texts.append(target_text)

                # Tính PPL nếu cần
                if calculate_ppl:
                    input_ids = batch['input_ids'][i].clone()
                    input_text = model.tokenizer.decode(input_ids, skip_special_tokens=False)
                    ppl = model.calculate_perplexity(input_text, target_text)
                    ppls.append(ppl)

                sample_idx += 1

    if gpu_monitor:
        gpu_monitor.log_stats("evaluation_end")

    # Tính các metrics chuẩn (BLEU, ROUGE, F1)
    refs = [[ref.lower().split()] for ref in target_texts]
    hyps = [hyp.lower().split() for hyp in output_texts]

    # BLEU scores
    smoothie = SmoothingFunction().method1
    bleu1 = sum([sentence_bleu([ref[0]], hyp, weights=(1, 0, 0, 0), smoothing_function=smoothie)
                 for ref, hyp in zip(refs, hyps)]) / len(refs)
    bleu4 = sum([sentence_bleu([ref[0]], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
                 for ref, hyp in zip(refs, hyps)]) / len(refs)

    # ROUGE scores
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

    # Tính Unigram F1 (so với target response)
    unigram_f1_scores = []
    for output, target in zip(output_texts, target_texts):
        f1 = calculate_unigram_f1(output, target)
        unigram_f1_scores.append(f1)
    avg_unigram_f1 = sum(unigram_f1_scores) / len(unigram_f1_scores) if unigram_f1_scores else 0.0

    # QUAN TRỌNG: Tính Knowledge F1 CHÍNH XÁC (so với gold knowledge)
    kf1_scores = []
    no_knowledge_count = 0

    for output, gold_knowledge in zip(output_texts, gold_knowledges):
        if gold_knowledge and gold_knowledge != "no_passages_used":
            kf1 = calculate_knowledge_f1(output, gold_knowledge)
            kf1_scores.append(kf1)
        else:
            # Trường hợp không có gold knowledge
            no_knowledge_count += 1
            kf1_scores.append(0.0)

    avg_kf1 = sum(kf1_scores) / len(kf1_scores) if kf1_scores else 0.0

    # Log thông tin về gold knowledge
    logger.info(f"Số mẫu có gold knowledge: {len(kf1_scores) - no_knowledge_count}/{len(kf1_scores)}")
    logger.info(f"Số mẫu không có gold knowledge: {no_knowledge_count}")

    # Tính trung bình PPL
    avg_ppl = sum(ppls) / len(ppls) if ppls else None

    # Tổng hợp kết quả
    results = {
        'bleu1': bleu1 * 100,
        'bleu4': bleu4 * 100,
        'rouge1': rouge_scores['rouge-1']['f'] * 100,
        'rouge2': rouge_scores['rouge-2']['f'] * 100,
        'rougeL': rouge_scores['rouge-l']['f'] * 100,
        'unigram_f1': avg_unigram_f1 * 100,
        'knowledge_f1': avg_kf1 * 100,  # Đây là KF1 CHÍNH XÁC
        'ppl': avg_ppl,
        'samples_with_knowledge': len(kf1_scores) - no_knowledge_count,
        'samples_without_knowledge': no_knowledge_count
    }

    # Hiển thị kết quả
    logger.info("=" * 60)
    logger.info("KẾT QUẢ ĐÁNH GIÁ (với Knowledge F1 chính xác):")
    logger.info("=" * 60)
    for metric, value in results.items():
        if isinstance(value, (int, float)) and metric not in ['samples_with_knowledge', 'samples_without_knowledge']:
            if metric == 'ppl':
                logger.info(f"{metric:15s}: {value:.2f}")
            else:
                logger.info(f"{metric:15s}: {value:.2f}%")
    logger.info("=" * 60)

    # Phân tích sự khác biệt giữa F1 và Knowledge F1
    logger.info("\n📊 PHÂN TÍCH CHI TIẾT:")
    logger.info(f"Unigram F1 (so với response): {avg_unigram_f1 * 100:.2f}%")
    logger.info(f"Knowledge F1 (so với gold knowledge): {avg_kf1 * 100:.2f}%")

    if avg_kf1 < avg_unigram_f1 * 0.7:  # Nếu KF1 thấp hơn đáng kể
        logger.warning("⚠️ Knowledge F1 thấp - mô hình có thể đang hallucinate!")
        logger.info("Model sinh response giống ground truth nhưng không dùng đúng tri thức")
    elif avg_kf1 > avg_unigram_f1:
        logger.info("✅ Knowledge F1 cao - mô hình sử dụng tốt tri thức được cung cấp")

    # Lưu kết quả chi tiết nếu cần
    if output_file:
        # Lưu metrics tổng hợp
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Lưu chi tiết một số mẫu để phân tích
        detailed_file = output_file.replace('.json', '_detailed_analysis.json')
        detailed_results = []

        for i in range(min(10, len(output_texts))):
            detailed_results.append({
                'sample_id': i + 1,
                'prediction': output_texts[i],
                'ground_truth_response': target_texts[i],
                'gold_knowledge': gold_knowledges[i] if i < len(gold_knowledges) else "",
                'unigram_f1_vs_response': unigram_f1_scores[i] * 100,
                'knowledge_f1_vs_gold': kf1_scores[i] * 100 if i < len(kf1_scores) else 0
            })

        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Đã lưu phân tích chi tiết vào: {detailed_file}")

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
        max_grad_norm=1.0,
        gpu_monitor=None
):
    """Huấn luyện mô hình GenKS cải tiến với theo dõi GPU"""

    accelerator = Accelerator(gradient_accumulation_steps=accumulation_steps)

    logger.info(f"Đang huấn luyện mô hình với {len(train_data)} mẫu trong {epochs} epochs")

    if gpu_monitor:
        gpu_monitor.log_stats("training_start", {"epochs": epochs, "batch_size": batch_size})

    train_dataset = ImprovedMultiSpanGENKSData(
        train_data,
        model.tokenizer,
        context_len=512,
        knowledge_len=256,
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
        num_workers=4
    )

    total_steps = len(train_dataloader) * epochs // accumulation_steps

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

    model.model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model.model, optimizer, train_dataloader, scheduler
    )

    best_eval_result = 0
    epoch_times = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        accelerator.wait_for_everyone()

        logger.info(f"Epoch {epoch + 1}/{epochs}")

        if gpu_monitor:
            gpu_monitor.log_stats(f"epoch_{epoch + 1}_start", {"epoch": epoch + 1})

        model.model.train()
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
        running_loss = 0
        batch_times = []

        for step, batch in enumerate(progress_bar):
            batch_start_time = time.time()

            if gpu_monitor and step % 50 == 0:
                gpu_monitor.log_stats(f"epoch_{epoch + 1}_batch_{step}",
                                      {"batch": step, "loss": running_loss / (step + 1) if step > 0 else 0})

            with accelerator.accumulate(model.model):
                outputs = model.model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            running_loss += loss.item()
            avg_batch_time = np.mean(batch_times)
            progress_bar.set_description(
                f"Loss: {running_loss / (step + 1):.4f} | Batch time: {avg_batch_time:.2f}s"
            )

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        if gpu_monitor:
            gpu_monitor.log_stats(f"epoch_{epoch + 1}_end", {
                "epoch": epoch + 1,
                "epoch_time": epoch_time,
                "avg_batch_time": np.mean(batch_times),
                "final_loss": running_loss / len(train_dataloader)
            })

        logger.info(f"Epoch {epoch + 1} hoàn thành trong {str(timedelta(seconds=int(epoch_time)))}")

        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model.model)
            unwrapped_model.save_pretrained(f"{output_dir}/epoch_{epoch + 1}")
            model.tokenizer.save_pretrained(f"{output_dir}/epoch_{epoch + 1}")

        if eval_data:
            eval_start_time = time.time()
            model.model = accelerator.unwrap_model(model.model)

            eval_results = evaluate_enhanced_genks(
                model=model,
                eval_data=eval_data,
                output_file=f"{output_dir}/eval_results_epoch_{epoch + 1}.json",
                batch_size=batch_size,
                calculate_ppl=True,
                gpu_monitor=gpu_monitor
            )

            eval_time = time.time() - eval_start_time
            logger.info(f"Đánh giá hoàn thành trong {str(timedelta(seconds=int(eval_time)))}")

            # Sử dụng Unigram F1 và BLEU4 để đánh giá mô hình tốt nhất
            current_metric = eval_results.get('unigram_f1', 0) * 0.5 + eval_results.get('bleu4', 0) * 0.5

            if current_metric > best_eval_result:
                best_eval_result = current_metric

                if accelerator.is_main_process:
                    logger.info(
                        f"Tìm thấy mô hình tốt nhất ở epoch {epoch + 1} với Combined Score: {current_metric:.2f}")
                    model.save(f"{output_dir}/best_model")

            model.model = accelerator.prepare(model.model)

    if accelerator.is_main_process:
        model.save(f"{output_dir}/final_model")

    total_training_time = sum(epoch_times)
    logger.info(f"\nTổng thời gian huấn luyện: {str(timedelta(seconds=int(total_training_time)))}")
    logger.info(f"Thời gian trung bình mỗi epoch: {str(timedelta(seconds=int(np.mean(epoch_times))))}")

    return model


def main():
    """Hàm chính để chạy quá trình huấn luyện và đánh giá với Unigram F1"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Khởi tạo bộ theo dõi GPU
    gpu_monitor = GPUMonitor(device='cuda')
    gpu_monitor.start_monitoring()

    total_start_time = time.time()

    # Khởi tạo mô hình
    logger.info("Đang khởi tạo mô hình GenKS với RAG...")
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

    gpu_monitor.log_stats("model_initialized")

    # Tải dữ liệu
    logger.info("Đang tải dữ liệu...")
    train_data = json.load(open('/kaggle/input/wizard/train.json'))
    valid_data = json.load(open('/kaggle/input/wizard/valid_seen.json'))
    test_seen_data = json.load(open('/kaggle/input/wizard/test_seen.json'))
    test_unseen_data = json.load(open('/kaggle/input/wizard/test_unseen.json'))

    gpu_monitor.log_stats("data_loaded", {
        "train_samples": len(train_data),
        "valid_samples": len(valid_data),
        "test_seen_samples": len(test_seen_data),
        "test_unseen_samples": len(test_unseen_data)
    })

    # Xây dựng corpus
    logger.info("Đang xây dựng corpus...")
    corpus_build_start = time.time()

    corpus = []
    for i, example in enumerate(train_data):
        if 'knowledge' in example:
            for title, sentences in example['knowledge'].items():
                for j, sentence in enumerate(sentences):
                    doc_id = f"doc_{i}_{j}"
                    corpus.append((doc_id, sentence, title))

    logger.info(f"Corpus chứa {len(corpus)} tài liệu")

    model.build_corpus_index(corpus, cache_path='/kaggle/working/ckpt/embeddings_cache.pt')

    corpus_build_time = time.time() - corpus_build_start
    logger.info(f"Xây dựng corpus hoàn thành trong {str(timedelta(seconds=int(corpus_build_time)))}")
    gpu_monitor.log_stats("corpus_built", {"corpus_time": corpus_build_time})

    # Huấn luyện mô hình
    logger.info("Bắt đầu quá trình huấn luyện...")
    training_start_time = time.time()

    model = train_enhanced_genks(
        model=model,
        train_data=train_data,
        eval_data=valid_data,
        output_dir='/kaggle/working/ckpt/enhanced_genks',
        epochs=3,
        batch_size=4,
        accumulation_steps=8,
        learning_rate=2e-5,
        warmup_steps=1000,
        gpu_monitor=gpu_monitor
    )

    training_time = time.time() - training_start_time
    logger.info(f"Huấn luyện hoàn thành trong {str(timedelta(seconds=int(training_time)))}")
    gpu_monitor.log_stats("training_completed", {"training_time": training_time})

    # Đánh giá trên test seen
    logger.info("Đánh giá trên tập test seen...")
    test_seen_start = time.time()

    results_seen = evaluate_enhanced_genks(
        model=model,
        eval_data=test_seen_data,
        output_file='/kaggle/working/ckpt/enhanced_genks/results_seen.json',
        batch_size=8,
        gpu_monitor=gpu_monitor
    )

    test_seen_time = time.time() - test_seen_start
    gpu_monitor.log_stats("test_seen_completed", {
        "test_time": test_seen_time,
        "results": results_seen
    })

    # Đánh giá trên test unseen
    logger.info("Đánh giá trên tập test unseen...")
    test_unseen_start = time.time()

    results_unseen = evaluate_enhanced_genks(
        model=model,
        eval_data=test_unseen_data,
        output_file='/kaggle/working/ckpt/enhanced_genks/results_unseen.json',
        batch_size=8,
        gpu_monitor=gpu_monitor
    )

    test_unseen_time = time.time() - test_unseen_start
    gpu_monitor.log_stats("test_unseen_completed", {
        "test_time": test_unseen_time,
        "results": results_unseen
    })

    total_time = time.time() - total_start_time

    # Lưu báo cáo
    monitoring_report_path = '/kaggle/working/ckpt/enhanced_genks/gpu_monitoring_report.json'
    gpu_monitor.save_monitoring_report(monitoring_report_path)

    time_report = {
        "total_execution_time": {
            "seconds": total_time,
            "formatted": str(timedelta(seconds=int(total_time)))
        },
        "corpus_building_time": {
            "seconds": corpus_build_time,
            "formatted": str(timedelta(seconds=int(corpus_build_time)))
        },
        "training_time": {
            "seconds": training_time,
            "formatted": str(timedelta(seconds=int(training_time)))
        },
        "evaluation_time": {
            "test_seen": {
                "seconds": test_seen_time,
                "formatted": str(timedelta(seconds=int(test_seen_time)))
            },
            "test_unseen": {
                "seconds": test_unseen_time,
                "formatted": str(timedelta(seconds=int(test_unseen_time)))
            }
        },
        "results": {
            "test_seen": results_seen,
            "test_unseen": results_unseen
        },
        "timestamp": datetime.now().isoformat()
    }

    time_report_path = '/kaggle/working/ckpt/enhanced_genks/time_report.json'
    with open(time_report_path, 'w', encoding='utf-8') as f:
        json.dump(time_report, f, indent=2, ensure_ascii=False)

    # Hiển thị kết quả cuối cùng
    logger.info("=" * 80)
    logger.info("KẾT QUẢ TỔNG HỢP VỚI UNIGRAM F1")
    logger.info("=" * 80)

    logger.info(f"\n⏱️ THỜI GIAN THỰC THI:")
    logger.info(f"  - Tổng thời gian: {time_report['total_execution_time']['formatted']}")
    logger.info(f"  - Xây dựng corpus: {time_report['corpus_building_time']['formatted']}")
    logger.info(f"  - Huấn luyện: {time_report['training_time']['formatted']}")
    logger.info(f"  - Đánh giá test_seen: {time_report['evaluation_time']['test_seen']['formatted']}")
    logger.info(f"  - Đánh giá test_unseen: {time_report['evaluation_time']['test_unseen']['formatted']}")

    if torch.cuda.is_available():
        logger.info(f"\n💻 THÔNG TIN GPU:")
        logger.info(f"  - GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  - Peak Memory: {gpu_monitor.peak_memory:.2f} GB")

        current_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"  - Current Memory: {current_memory:.2f}/{total_memory:.2f} GB")

    logger.info("\n📊 KẾT QUẢ TRÊN TẬP TEST SEEN:")
    logger.info("-" * 40)
    for metric, value in results_seen.items():
        if value is not None:
            if metric == 'ppl':
                logger.info(f"  - {metric:12s}: {value:.2f}")
            else:
                logger.info(f"  - {metric:12s}: {value:.2f}%")

    if 'unigram_f1' in results_seen and 'bleu4' in results_seen:
        combined_score_seen = (results_seen['unigram_f1'] + results_seen['bleu4']) / 2
        logger.info(f"  - {'Combined F1+BLEU4':12s}: {combined_score_seen:.2f}%")

    logger.info("\n📊 KẾT QUẢ TRÊN TẬP TEST UNSEEN:")
    logger.info("-" * 40)
    for metric, value in results_unseen.items():
        if value is not None:
            if metric == 'ppl':
                logger.info(f"  - {metric:12s}: {value:.2f}")
            else:
                logger.info(f"  - {metric:12s}: {value:.2f}%")

    if 'unigram_f1' in results_unseen and 'bleu4' in results_unseen:
        combined_score_unseen = (results_unseen['unigram_f1'] + results_unseen['bleu4']) / 2
        logger.info(f"  - {'Combined F1+BLEU4':12s}: {combined_score_unseen:.2f}%")

    logger.info("\n🔍 SO SÁNH CÁC METRICS F1:")
    logger.info("-" * 40)
    if 'unigram_f1' in results_seen and 'kf1' in results_seen:
        logger.info("Test Seen:")
        logger.info(f"  - Unigram F1: {results_seen['unigram_f1']:.2f}%")
        logger.info(f"  - Knowledge F1: {results_seen['kf1']:.2f}%")
        diff_seen = results_seen['unigram_f1'] - results_seen['kf1']
        logger.info(f"  - Chênh lệch: {diff_seen:+.2f}%")

    if 'unigram_f1' in results_unseen and 'kf1' in results_unseen:
        logger.info("\nTest Unseen:")
        logger.info(f"  - Unigram F1: {results_unseen['unigram_f1']:.2f}%")
        logger.info(f"  - Knowledge F1: {results_unseen['kf1']:.2f}%")
        diff_unseen = results_unseen['unigram_f1'] - results_unseen['kf1']
        logger.info(f"  - Chênh lệch: {diff_unseen:+.2f}%")

    logger.info("\n📈 BẢNG TỔNG HỢP TẤT CẢ METRICS:")
    logger.info("-" * 60)
    logger.info(f"{'Metric':<15} {'Test Seen':>12} {'Test Unseen':>12} {'Difference':>12}")
    logger.info("-" * 60)

    metrics_order = ['bleu1', 'bleu4', 'rouge1', 'rouge2', 'rougeL', 'unigram_f1', 'kf1', 'ppl']
    for metric in metrics_order:
        if metric in results_seen and metric in results_unseen:
            seen_val = results_seen[metric]
            unseen_val = results_unseen[metric]
            if seen_val is not None and unseen_val is not None:
                diff = seen_val - unseen_val
                if metric == 'ppl':
                    logger.info(f"{metric:<15} {seen_val:>12.2f} {unseen_val:>12.2f} {diff:>+12.2f}")
                else:
                    logger.info(f"{metric:<15} {seen_val:>11.2f}% {unseen_val:>11.2f}% {diff:>+11.2f}%")

    logger.info("-" * 60)

    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Báo cáo theo dõi GPU đã lưu tại: {monitoring_report_path}")
    logger.info(f"✅ Báo cáo thời gian đã lưu tại: {time_report_path}")
    logger.info(f"✅ Kết quả Test Seen đã lưu tại: /kaggle/working/ckpt/enhanced_genks/results_seen.json")
    logger.info(f"✅ Kết quả Test Unseen đã lưu tại: /kaggle/working/ckpt/enhanced_genks/results_unseen.json")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()