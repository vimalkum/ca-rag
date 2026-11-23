import re, random, json
from string import Template
from datetime import date, timedelta
import streamlit as st
import time
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from modules.extract_ontology import OntologyExtractor
from modules.KG_extractor import SemanticExtractor

file_path = "./PolicyFlow/modules/Dataset/DSA_Ontology.ttl" 
kg_file = "./PolicyFlow/modules/Dataset/finance_kg.ttl"
model_path = 'all-MiniLM-L6-v2'


class DSAGenerator:
    def __init__(self, titles, desc, rag_chain, prompt_manager, llm, retriever, oflag, kgflag):
        self.titles = titles
        self.desc = desc
        self.rag_chain = rag_chain
        self.prompt_manager = prompt_manager
        self.retriever = retriever
        self.out = []
        self.ext_out = []
        self.oflag=oflag
        self.kgflag = kgflag
        self.model_num_params: float = 3e9  # e.g., 7e9
        if self.oflag:
            self.ontology_extractor = OntologyExtractor(file_path)
        else:
            self.ontology_extractor = ''
        if self.kgflag:
            self.kg_extractor = SemanticExtractor(model_path)
        else:
            self.kg_extractor = ''
        
    @staticmethod
    def random_date(start, end):
        delta = end - start
        int_delta = delta.days
        random_day = random.randrange(int_delta)
        return start + timedelta(days=random_day)
    
    @staticmethod
    def extract_text(llm_text):
        """
        Extracts the output portion from LLM-generated text by removing JSON data
        and keeping only the textual explanation/answer.
        
        Args:
            llm_text (str): The full LLM response text
            
        Returns:
            str: The extracted output text with JSON removed
        """
        # Remove JSON arrays/objects
        text = re.sub(r'\[{.*?}\]', '', llm_text, flags=re.DOTALL)
        
        # Remove the initial label/question part (e.g., "Third-Party Use Restrictions :")
        text = re.sub(r'^[^:]+:\s*', '', text.strip())

        # Remove all "End of..." patterns (case insensitive)
        text = re.sub(r'End of (Output|Response|Task|Conversation|Session|Engagement|Interaction|Discussion|Talk|Dialogue)\.?\s*', '', text, flags=re.IGNORECASE)
        
        # Remove repetitive correction/final answer patterns
        text = re.sub(r'(The final answer is:|This response is incorrect|The correct response is:|Here is the corrected response:|I will make sure to follow)[^\n]*', '', text, flags=re.IGNORECASE)
        
        # Remove metadata sections like "Type/Class Description:", "Properties:", "##Output should be"
        text = re.sub(r'\*\s*Type/Class Description:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'##\s*Properties:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'Properties:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'##Output should be.*?(?=\n|\Z)', '', text, flags=re.IGNORECASE | re.DOTALL)
    
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


    @staticmethod
    def remove_incomplete_sentences(paragraph):
        sentences = re.split(r'(?<=[.!?])(\s+)', paragraph)
        cleaned_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence, space = sentences[i], sentences[i + 1]
            if re.search(r'[.!?]$', sentence.strip()):
                cleaned_sentences.append(sentence + space)
        return "".join(cleaned_sentences)

    @staticmethod
    def remove_repeated_phrases(paragraph):
        seen_phrases = set()
        cleaned_lines = []
        for line in paragraph.splitlines(keepends=True):
            clean_line = re.sub(r'\s+', ' ', line.strip())
            if clean_line and clean_line not in seen_phrases:
                seen_phrases.add(clean_line)
                cleaned_lines.append(line)
        return "".join(cleaned_lines)

    def clean_paragraph(self, paragraph, check):
        paragraph = paragraph.split('Human:')[0]
        paragraph = paragraph.split('System:')[0]
        paragraph = paragraph.split('Output:')[0]
        paragraph = paragraph.split('Note:')[0]
        paragraph = paragraph.split('python')[0]
        paragraph = self.extract_text(paragraph)
        paragraph = self.remove_repeated_phrases(paragraph)
        
        if check != 2:
            paragraph = self.remove_incomplete_sentences(paragraph)
        return paragraph

    @staticmethod
    def extract_bracketed_values(text):
        return str(re.findall(r'\[(.*?)\]', text))

    @staticmethod
    def remove_duplicates(instr):
        tstr = instr[1:len(instr)-2].split(', ')
        seen = set()
        unique = []
        for w in tstr:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return str(unique)
    def extract_section_headings(self, text: str) -> list[str]:
        """
        Extract section headings from a block of text containing numbered headings.
    
        Args:
            text (str): Input text containing section headings.
    
        Returns:
            list[str]: A list of extracted section heading titles.
        """
        # Regex to match patterns like: 1. Heading text
        pattern = r'\d+\.\s*([A-Za-z][A-Za-z\s,()-]+)'
    
        matches = re.findall(pattern, text)
    
        # Clean up whitespace and trailing commas
        headings = [m.strip().rstrip(',') for m in matches]
        headings = list(set(headings))
        
        return headings

    def _safe_split_generated(self, answer_text: str) -> str:
        if not answer_text:
            return ""
        if "Generated Output:" in answer_text:
            return answer_text.split("Generated Output:", 1)[1].rstrip()
        return answer_text.rstrip()

    def _extract_usage_tokens(self, output_obj: dict):
        """
        Best-effort extraction of token usage from different LangChain/LLM providers.
        Returns (prompt_tokens, completion_tokens, total_tokens) or (None, None, None).
        """
        if not isinstance(output_obj, dict):
            return (None, None, None)

        # Common patterns
        usage = output_obj.get("usage") or output_obj.get("token_usage")
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            tt = usage.get("total_tokens")
            if any(x is not None for x in (pt, ct, tt)):
                return (pt, ct, tt)

        # LangChain sometimes stores provider payload under llm_output
        llm_output = output_obj.get("llm_output")
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage") or llm_output.get("usage")
            if isinstance(usage, dict):
                pt = usage.get("prompt_tokens")
                ct = usage.get("completion_tokens")
                tt = usage.get("total_tokens")
                if any(x is not None for x in (pt, ct, tt)):
                    return (pt, ct, tt)

        return (None, None, None)

    def _count_tokens_fallback(self, text: str) -> int:
        """
        Tries tiktoken; if not available, uses a rough heuristic.
        """
        if not text:
            return 0
        try:
            import tiktoken
            # Use a general encoding if you don't know the exact model
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Very rough fallback: ~1.3 tokens/word
            return int(math.ceil(len(text.split()) * 1.3))

    def _estimate_flops(self, prompt_tokens: int, gen_tokens: int) -> float:
        """
        Returns estimated FLOPs for the request, or None if params unknown.
        """
        if not self.model_num_params:
            return None
        # Simple inference proxy (forward-only):
        # FLOPs ≈ 2 * params * (prompt_tokens + gen_tokens)
        return 2.0 * float(self.model_num_params) * float((prompt_tokens or 0) + (gen_tokens or 0))
    def _pname_pvalue_from_desc(self, title: str):
        """
        Convert desc[title] like:
          { "propLabel": {"comment": "...", ...} }
        into:
          (pname: str, pvalue: str)
        """
        d = self.desc.get(title, {})
        if not d or not isinstance(d, dict):
            return title, ""
    
        # take first/best property entry
        prop_name = next(iter(d.keys()), None)
        if not prop_name:
            return title, ""
    
        meta = d.get(prop_name, {})
        if isinstance(meta, dict):
            hint = meta.get("comment", "") or meta.get("definition", "") or ""
        else:
            hint = str(meta) if meta is not None else ""
    
        return prop_name, hint

    def generate(self, user_inputs, create=False):
        tparty = 'Yes'
        flag, third = True, False
        out, ext_out, prmpt_dict, docs = dict(), [], dict(), dict()

        # ---- Metrics containers ----
        metrics = {
            "calls": [],   # list of per-invoke metrics
            "totals": {
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "total_tokens": 0,
                "latency_s": 0.0
            }
        }

        # ---------- 1) Title generation call ----------
        prmpt = self.prompt_manager.get_title_prompt(ctext=user_inputs['user_input'])

        t0 = time.perf_counter()
        output_obj = self.rag_chain.invoke({"input": prmpt}, config={"max_tokens": 512, 'repetition_penalty': 1.8})
        t1 = time.perf_counter()

        latency_s = t1 - t0
        answer_text = output_obj.get("answer", "")
        gen_text = self._safe_split_generated(answer_text)

        pt, ct, tt = self._extract_usage_tokens(output_obj)
        if pt is None:
            pt = self._count_tokens_fallback(prmpt)
        if ct is None:
            ct = self._count_tokens_fallback(gen_text)
        if tt is None:
            tt = pt + ct

        flops = self._estimate_flops(pt, ct)
        tflops = (flops / latency_s / 1e12) if (flops is not None and latency_s > 0) else None

        metrics["calls"].append({
            "stage": "title_generation",
            "prompt_tokens": pt,
            "generated_tokens": ct,
            "total_tokens": tt,
            "latency_s": latency_s,
            "latency_ms": latency_s * 1000.0,
            "tflops_est": tflops
        })
        metrics["totals"]["prompt_tokens"] += pt
        metrics["totals"]["generated_tokens"] += ct
        metrics["totals"]["total_tokens"] += tt
        metrics["totals"]["latency_s"] += latency_s

        self.titles = self.extract_section_headings(gen_text)
        st.session_state.generated_titles = self.titles
        if self.oflag:
            self.desc, concepts_by_title = self.ontology_extractor.extract_info(self.titles)
        #print(self.desc)

        with st.status("Generating DSA", expanded=True) as status:
            for t in self.titles:
                if self.oflag:
                    title_key = t.rstrip()
                
                    # ✅ pname/pvalue extracted correctly
                    pname, pvalue = self._pname_pvalue_from_desc(title_key)
                
                    # concepts top-3 (already structured)
                    title_concepts = concepts_by_title.get(title_key, [])
                    classes = json.dumps(title_concepts, ensure_ascii=False)
                else:
                    pname, pvalue = t.rstrip(), ""
                    classes = ""

                print(pname, pvalue, classes)
                if self.kgflag:
                    t3 = (
                    "Use the below knowledge graph to understand more about the data usage policy statements required for this title. "
                    "The knowledge graph is given as a list of triples with (subject, predicate, object) format. \nKnowledge Graph: "
                    + str(self.kg_extractor.extract_relevant_triples(kg_file, t.rstrip()))
                )
                else:
                    t3=''
                print(t3)
                st.write('Generating for ' + t)
                prmpt = self.prompt_manager.get_prompt(
                    pname=pname,
                    hint=pvalue,
                    classes=classes,
                    index=1,
                    ctext=user_inputs['user_input'],
                    title=t.rstrip(),
                    triples=t3
                )

                prmpt_dict[t.rstrip()] = prmpt

                # ---------- 2) Per-title generation call ----------
                t0 = time.perf_counter()
                output_obj = self.rag_chain.invoke({"input": prmpt}, config={"max_tokens": 512, 'repetition_penalty': 1.8})
                t1 = time.perf_counter()

                latency_s = t1 - t0
                docs[t.rstrip()] = self.retriever.get_relevant_documents(prmpt)

                answer_text = output_obj.get("answer", "")
                gen_text = self._safe_split_generated(answer_text)
                clean_out = self.clean_paragraph(gen_text, 1)

                pt, ct, tt = self._extract_usage_tokens(output_obj)
                if pt is None:
                    pt = self._count_tokens_fallback(prmpt)
                if ct is None:
                    ct = self._count_tokens_fallback(gen_text)
                if tt is None:
                    tt = pt + ct

                flops = self._estimate_flops(pt, ct)
                tflops = (flops / latency_s / 1e12) if (flops is not None and latency_s > 0) else None

                metrics["calls"].append({
                    "stage": "dsa_generation",
                    "title": t.rstrip(),
                    "prompt_tokens": pt,
                    "generated_tokens": ct,
                    "total_tokens": tt,
                    "latency_s": latency_s,
                    "latency_ms": latency_s * 1000.0,
                    "tflops_est": tflops
                })
                metrics["totals"]["prompt_tokens"] += pt
                metrics["totals"]["generated_tokens"] += ct
                metrics["totals"]["total_tokens"] += tt
                metrics["totals"]["latency_s"] += latency_s

                print(clean_out)

                ext_out = ''
                st.write('Generated for ' + t)
                out[t.rstrip()] = clean_out
                #print(out)

            status.update(label="Generated DSA", state="complete", expanded=False)

        self.out, self.ext_out = out, ext_out

        # store metrics so you can display them in Streamlit or log them
        self.metrics = metrics

        # Return metrics as an extra output (optional but handy)
        return self.out, self.ext_out, prmpt_dict, docs, metrics


