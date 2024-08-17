import os
import json
import esprima
from typing import Dict, List
import tarfile
import tempfile
import shutil
import requests
import networkx as nx
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SDKAnalyzer:
    def __init__(self, package_name: str, version: str = 'latest'):
        self.package_name = package_name
        self.version = version
        self.package_info = None
        self.temp_dir = tempfile.mkdtemp()
        self.extracted_path = None
        self.analysis_result = {
            "name": package_name,
            "version": version,
            "description": "",
            "main_file": "",
            "functions": [],
            "classes": [],
            "dependencies": {},
            "dev_dependencies": {},
            "ai_summary": "",
            "potential_use_cases": []
        }
        self.knowledge_graph = nx.DiGraph()
        self.mongo_client = MongoClient('mongodb://localhost:27017/')
        self.db = self.mongo_client['sdk_analyzer']
        
        # Use smaller models
        self.tokenizer = None
        self.model = None
        self.summarizer = None
        self.text_generator = None

    def _load_models(self):
        try:
            # Use DistilBERT instead of CodeBERT (smaller and faster)
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
            
            # Use smaller summarization and text generation models
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", max_length=60, min_length=30)
            self.text_generator = pipeline("text-generation", model="distilgpt2", max_length=50)
            
            logger.info("AI models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading AI models: {str(e)}")
            raise

    def download_and_extract(self):
        self._fetch_package_info()
        if self.version == 'latest':
            self.version = self.package_info['dist-tags']['latest']
        
        tarball_url = self.package_info['versions'][self.version]['dist']['tarball']
        logger.info(f"Downloading from URL: {tarball_url}")
        response = requests.get(tarball_url)
        
        if response.status_code != 200:
            raise ValueError(f"Failed to download package: HTTP {response.status_code}")
        
        tgz_file = os.path.join(self.temp_dir, f"{self.package_name}-{self.version}.tgz")
        
        with open(tgz_file, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Extracting {tgz_file}")
        try:
            with tarfile.open(tgz_file, "r:gz") as tar:
                tar.extractall(path=self.temp_dir)
        except tarfile.ReadError:
            raise ValueError("Downloaded file is not a valid gzip file. The package might not exist or the version might be incorrect.")
        
        self.extracted_path = self.temp_dir
        logger.info(f"Extracted to {self.extracted_path}")
        self._find_package_root()

    def _find_package_root(self):
        logger.info(f"Searching for package.json in {self.extracted_path}")
        for root, dirs, files in os.walk(self.extracted_path):
            logger.debug(f"Checking directory: {root}")
            logger.debug(f"Files: {files}")
            if 'package.json' in files:
                self.extracted_path = root
                logger.info(f"Found package.json in {self.extracted_path}")
                return
        
        # If we couldn't find package.json, log the directory structure
        logger.error("Could not find package.json. Directory structure:")
        for root, dirs, files in os.walk(self.extracted_path):
            level = root.replace(self.extracted_path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            logger.error(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                logger.error(f"{subindent}{f}")
        
        raise ValueError(f"Unable to find package.json in the extracted files. Extraction path: {self.extracted_path}")

    def _fetch_package_info(self):
        url = f"https://registry.npmjs.org/{self.package_name}"
        logger.info(f"Fetching package info from {url}")
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch package info: HTTP {response.status_code}")
        self.package_info = response.json()
        logger.info("Successfully fetched package info")

    def _analyze_package_json(self):
        package_json_path = os.path.join(self.extracted_path, 'package.json')
        logger.info(f"Analyzing package.json at {package_json_path}")
        
        # Log the contents of the directory
        logger.debug(f"Contents of {self.extracted_path}:")
        for item in os.listdir(self.extracted_path):
            logger.debug(f"  {item}")
        
        if not os.path.exists(package_json_path):
            logger.error(f"package.json not found in {self.extracted_path}")
            raise ValueError(f"package.json not found in {self.extracted_path}")
        
        try:
            with open(package_json_path, 'r') as f:
                package_json = json.load(f)
        except IOError as e:
            logger.error(f"Error reading package.json: {str(e)}")
            raise ValueError(f"Error reading package.json: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing package.json: {str(e)}")
            raise ValueError(f"Error parsing package.json: {str(e)}")
        
        self.analysis_result["description"] = package_json.get("description", "")
        self.analysis_result["main_file"] = package_json.get("main", "index.js")
        self.analysis_result["dependencies"] = package_json.get("dependencies", {})
        self.analysis_result["dev_dependencies"] = package_json.get("devDependencies", {})
        logger.info("Successfully analyzed package.json")

    def _analyze_main_file(self):
        main_file_path = os.path.join(self.extracted_path, self.analysis_result["main_file"])
        logger.info(f"Analyzing main file at {main_file_path}")
        
        if not os.path.exists(main_file_path):
            logger.warning(f"Main file {main_file_path} not found.")
            return

        try:
            with open(main_file_path, 'r') as f:
                content = f.read()
            logger.debug(f"Main file content (first 100 chars): {content[:100]}...")

            tree = esprima.parseModule(content, {'jsx': True, 'loc': True})
            self._extract_functions(tree)
            self._extract_classes(tree)
            self._generate_ai_summary(content)
            self._suggest_use_cases(content)
            logger.info("Successfully analyzed main file")
        except Exception as e:
            logger.error(f"Error analyzing main file: {str(e)}")
            raise

    def _generate_ai_summary(self, content):
        try:
            if not self.summarizer:
                return "AI summary generation is not available."
            
            simple_summary = f"This is a JavaScript package named {self.package_name}. " \
                             f"It contains {len(self.analysis_result['functions'])} functions and " \
                             f"{len(self.analysis_result['classes'])} classes."

            if len(content) > 500:
                content = content[:500]  # Limit content to first 500 characters
            summary = self.summarizer(content, max_length=60, min_length=30, do_sample=False)
            ai_summary = summary[0]['summary_text']

            self.analysis_result["ai_summary"] = ai_summary + " " + simple_summary
            logger.info("AI summary generated successfully")
        except Exception as e:
            logger.error(f"Error generating AI summary: {str(e)}")
            self.analysis_result["ai_summary"] = simple_summary

    def _suggest_use_cases(self, content):
        try:
            basic_use_cases = [
                f"Use {self.package_name} for authentication in Node.js applications.",
                f"Implement Google Sign-In in web applications using {self.package_name}.",
                f"Manage OAuth 2.0 flows with {self.package_name} in server-side applications."
            ]

            if not self.text_generator:
                self.analysis_result["potential_use_cases"] = basic_use_cases
                return

            prompt = f"Suggest a use case for a JavaScript package named {self.package_name}"
            ai_use_case = self.text_generator(prompt, max_length=50, num_return_sequences=1)
            
            combined_use_cases = basic_use_cases + [ai_use_case[0]['generated_text']]
            
            self.analysis_result["potential_use_cases"] = combined_use_cases
            logger.info(f"Generated {len(combined_use_cases)} potential use cases")
        except Exception as e:
            logger.error(f"Error suggesting use cases: {str(e)}")
            self.analysis_result["potential_use_cases"] = basic_use_cases

    def _extract_functions(self, tree):
        self.analysis_result["functions"] = []
        for node in tree.body:
            if node.type == 'FunctionDeclaration':
                self.analysis_result["functions"].append({
                    "name": node.id.name,
                    "params": [param.name for param in node.params],
                    "docstring": self._extract_docstring(node)
                })
            elif node.type == 'VariableDeclaration':
                for decl in node.declarations:
                    if decl.init and decl.init.type == 'FunctionExpression':
                        self.analysis_result["functions"].append({
                            "name": decl.id.name,
                            "params": [param.name for param in decl.init.params],
                            "docstring": self._extract_docstring(decl.init)
                        })

    def _extract_classes(self, tree):
        self.analysis_result["classes"] = []
        for node in tree.body:
            if node.type == 'ClassDeclaration':
                methods = []
                for method in node.body.body:
                    if method.type == 'MethodDefinition':
                        methods.append({
                            "name": method.key.name,
                            "params": [param.name for param in method.value.params],
                            "docstring": self._extract_docstring(method.value)
                        })
                self.analysis_result["classes"].append({
                    "name": node.id.name,
                    "methods": methods,
                    "docstring": self._extract_docstring(node)
                })

    def _extract_docstring(self, node):
        # This is a simple implementation. You might want to enhance this
        # to handle more complex JSDoc patterns.
        if node.leadingComments:
            return node.leadingComments[0].value
        return ""

    def create_knowledge_graph(self):
        self.knowledge_graph.add_node(self.package_name, type='package', version=self.version)

        for func in self.analysis_result['functions']:
            func_name = f"{self.package_name}.{func['name']}"
            self.knowledge_graph.add_node(func_name, type='function', params=func['params'], docstring=func['docstring'])
            self.knowledge_graph.add_edge(self.package_name, func_name, type='contains')

        for cls in self.analysis_result['classes']:
            cls_name = f"{self.package_name}.{cls['name']}"
            self.knowledge_graph.add_node(cls_name, type='class', docstring=cls['docstring'])
            self.knowledge_graph.add_edge(self.package_name, cls_name, type='contains')

            for method in cls['methods']:
                method_name = f"{cls_name}.{method['name']}"
                self.knowledge_graph.add_node(method_name, type='method', params=method['params'], docstring=method['docstring'])
                self.knowledge_graph.add_edge(cls_name, method_name, type='contains')

        for dep, version in self.analysis_result['dependencies'].items():
            self.knowledge_graph.add_node(dep, type='dependency', version=version)
            self.knowledge_graph.add_edge(self.package_name, dep, type='depends_on')

    def store_knowledge_graph(self):
        kg_collection = self.db['knowledge_graphs']
        kg_data = nx.node_link_data(self.knowledge_graph)
        kg_collection.update_one(
            {'name': self.package_name, 'version': self.version},
            {'$set': {'graph': kg_data}},
            upsert=True
        )

    def generate_embeddings(self):
        embeddings = {}
        for node, data in self.knowledge_graph.nodes(data=True):
            text = f"{node} {data.get('type', '')} {data.get('docstring', '')}"
            embedding = self.sentence_model.encode(text)
            embeddings[node] = embedding.tolist()

        embedding_collection = self.db['embeddings']
        embedding_collection.update_one(
            {'name': self.package_name, 'version': self.version},
            {'$set': {'embeddings': embeddings}},
            upsert=True
        )

    def analyze(self):
        try:
            self.download_and_extract()
            self._analyze_package_json()
            self._analyze_main_file()
            self.create_knowledge_graph()
            self.store_knowledge_graph()
            
            # Only load AI models if needed and not already loaded
            if not self.summarizer:
                self._load_models()
            
            self._generate_ai_summary(self.analysis_result['description'])
            self._suggest_use_cases(self.analysis_result['description'])
            
            return self.analysis_result
        except Exception as e:
            logger.exception("An error occurred during analysis")
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "extracted_path": self.extracted_path,
            }
        finally:
            # Clean up the temporary directory
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Temporary directory cleaned up")

def analyze_sdk(package_name: str, version: str = 'latest') -> Dict:
    analyzer = SDKAnalyzer(package_name, version)
    return analyzer.analyze()

def semantic_search(query: str, package_name: str, version: str = 'latest'):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['sdk_analyzer']
    embedding_collection = db['embeddings']
    
    # Retrieve embeddings
    result = embedding_collection.find_one({'name': package_name, 'version': version})
    if not result:
        return []

    embeddings = result['embeddings']

    # Generate query embedding
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = sentence_model.encode(query)

    # Compute similarities
    similarities = {}
    for node, embedding in embeddings.items():
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        similarities[node] = similarity

    # Sort by similarity
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Return top 5 results
    return sorted_results[:5]

if __name__ == "__main__":
    # Example usage
    result = analyze_sdk("axios", "0.21.1")
    print(json.dumps(result, indent=2))

    # Example semantic search
    search_results = semantic_search("How to make a GET request?", "axios", "0.21.1")
    print("Search results:", search_results)