from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
import glob

class DocumentLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    def load_documents(self):
        files = glob.glob(self.dataset_path + "/**/*", recursive=True)
        docs = []
        for file in files:
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file)
                pages = loader.load_and_split(self.text_splitter)
                docs.extend(pages)
        return docs

    @staticmethod
    def read_txt(file_path):
        with open(file_path, 'r') as f:
            return f.read().split('\n')

    @staticmethod
    def extract_desc(fname):
        desc = dict()
        with open(fname, 'r', encoding='UTF-8') as file:
            for line in file:
                temp = line.rstrip().split(':')
                if len(temp) > 1:
                    desc[temp[0]] = temp[1]
        return desc

    @staticmethod
    def extract_desc(fname):
        desc = dict()
        with open(fname, 'r', encoding='UTF-8') as file:
            while line := file.readline():
                temp = line.rstrip().split(':')
                desc[temp[0]]=temp[1]
        return desc
    

