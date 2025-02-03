import os
import pandas as pd
import json
import torch
import transformers
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from dotenv import load_dotenv

# loading env variables
load_dotenv()

conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};' \
               'SERVER=tmssql03.tmsdomain.com\\tmssql1;' \
               'DATABASE=Salesforce_DW;' \
               'UID=' + os.environ.get('db_user') + ';' \
               'PWD=' + os.environ.get('db_pass')
connection_url = URL.create('mssql+pyodbc', query={'odbc_connect': conn_str})
engine = create_engine(connection_url)

class contextManager:
    def __init__(self, context_folder, k1, beta, bert_weight, bm25_weight):
        self.root = context_folder
        self.bert_weight = bert_weight
        self.bm25_weight = bm25_weight

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # k1 controls saturation of term frequency
        self.k1 = k1
        # beta controls the degree of length normalization (i.e. it is multiplied by our doc length)
        self.beta = beta

        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = transformers.BertModel.from_pretrained('bert-base-uncased')

        self.load()
        self.embed()

    def load(self):
        self.datasets = {}
        self.data_dictionaries = {}
        self.prompt_library = {}
        self.definitions = {}

        # Updating Branch Definitions
        branch_query = '''SELECT 
        d.name,
        max(b.name) as main_name,
        string_agg(b.Name, ',')  as aliases,
        max(e.Name) as manager_name

        FROM [Salesforce_Internal].[dbo].[Branch] b
        inner join Salesforce_Internal.dbo.Division d
        on b.Branch_Code__c = d.Id
        left join Salesforce_Internal.dbo.Employee e
        on d.Division_Code_Manager__c = e.Id

        where d.Is_Active__c = 1

        group by d.Name'''

        branches = pd.read_sql(branch_query, engine)
        with open(f'{self.root}/definitions/branch_codes.json', 'r') as f:
            branch_definitions = json.load(f)
            for branch in branches.iterrows():
                branch_definitions['definitions'][branch[1]['name']] = {'definition': f'The {branch[1]['main_name']} branch', 'description': f'branch code {branch[1]['name']}, managed by {branch[1]['manager_name']}', 'aliases': branch[1]['aliases'].split(',')}

        with open(f'{self.root}/definitions/branch_codes.json', 'w+') as f:
            json.dump(branch_definitions, f)

        # Loading datasets and data dicts
        f_names = [i.split('.')[0] for i in os.listdir(f'{self.root}/data dictionaries')]
        for f_name in f_names:
            with open(f'{self.root}/queries/{f_name}.sql') as f:
                self.datasets[f_name] = pd.read_sql(f.read(), engine)
            with open(f'{self.root}/data dictionaries/{f_name}.json') as f:
                self.data_dictionaries[f_name] = json.load(f)    

        # Loading definitions
        definition_groups = os.listdir(f'{self.root}/definitions')
        for file_name in definition_groups:
            group = file_name.split('.')[0]
            with open(f'{self.root}/definitions/{file_name}', 'r') as f:
                group_definitions = json.load(f)
                self.definitions[group] = group_definitions

        # Loading prompts
        prompt_libraries = os.listdir(f'{self.root}/prompt library')
        for library in prompt_libraries:
            with open(f'{self.root}/prompt library/{library}', 'r') as f:
                prompts = json.load(f)
            library_name = library.split('.')[0]
            self.prompt_library[library_name] = prompts
        
        return
    
    def embed(self):
        # Embedding definitions
        self.definition_category_index = {}
        self.definition_index = {}
        for key in self.definitions:
            category_definitions = self.definitions[key]['definitions']
            item = f'''
            Title: {self.definitions[key]['title']}
            Description: {self.definitions[key]['description']}
            Definitions: {', '.join(category_definitions)}
            '''
            trimmed_item = item.strip()
            self.definition_category_index[key] = {'item': trimmed_item, 'bert_vec': self.get_bert_embedding(trimmed_item)}

            for definition in category_definitions:
                item = f'''
                Word: {definition}
                Definition: {category_definitions[definition]['definition']}
                Description: {category_definitions[definition]['description']}
                Aliases: {category_definitions[definition]['aliases']}
                '''
                if key not in self.definition_index:
                    self.definition_index[key] = {}
                trimmed_item = item.strip()
                
                self.definition_index[key][definition] = {'item': trimmed_item, 'bert_vec': self.get_bert_embedding(trimmed_item)}


        # Embedding prompts
        self.prompt_category_index = {}
        self.prompt_index = {}
        for key in self.prompt_library:
            category_prompts = self.prompt_library[key]['prompts']
            item = f'''
            Title: {self.prompt_library[key]['title']}
            Description: {self.prompt_library[key]['description']}
            Prompts: {', '.join(category_prompts)}
            '''
            trimmed_item = item.strip()
            self.prompt_category_index[key] = {'item': trimmed_item, 'bert_vec': self.get_bert_embedding(trimmed_item)}
            for prompt in category_prompts:
                item = f'''
                Prompt Name: {prompt}
                Description: {category_prompts[prompt]['description']}
                Aliases: {', '.join(category_prompts[prompt]['aliases'])}
                Relevant Functions: {', '.join(category_prompts[prompt]['relevant_functions'])}
                Example Prompt: {category_prompts[prompt]['example_prompt']}
                Example Solution: {category_prompts[prompt]['example_solution']}
                '''
                if key not in self.prompt_index:
                    self.prompt_index[key] = {}
                trimmed_item = item.strip()
                self.prompt_index[key][prompt] = {'item': trimmed_item, 'bert_vec': self.get_bert_embedding(trimmed_item)}

        self.get_bm25_embedding()
        return

    def get_bert_embedding(self, item):
        # BERT embedding
        with torch.no_grad():
            tokenized = self.tokenizer(item, return_tensors='pt', max_length=512, padding='max_length', truncation=True).to(self.device)
            tokens = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            embedded = self.model(tokens, attention_mask=attention_mask)['last_hidden_state'].squeeze().to(self.device)
            mask = attention_mask.T.expand(embedded.size()).float()
            summed = (embedded * mask).sum(dim=1)
            attention_count = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / attention_count

        return pooled
    
    def get_bm25_embedding(self):
        # embedding items
        vocab = []
        keys = []
        indices = {'definitions': (self.definition_index, self.definition_category_index), 'prompts': (self.prompt_index, self.prompt_category_index)}
        for index_key in indices:
            index = indices[index_key]
            for item_key in index[0]:
                category_item = index[1][item_key]['item']
                vocab.append(category_item)
                keys.append((index_key, item_key))

                items = index[0][item_key]
                vocab.extend(items[i]['item'] for i in items)
                keys.extend((index_key, item_key, i) for i in items)

        self.count_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2,4))
        self.doc_term_matrix = torch.tensor(self.count_vectorizer.fit_transform(vocab).toarray(), dtype=torch.float32).to(self.device)

        self.idf_vec = self.idf_vectorizer()
        tf_matrix = self.tf_vectorizer()

        for index, key in enumerate(keys):
            bm25_vec = tf_matrix[index]
            if len(key) == 2:
                # this is a category key
                if key[0] == 'definitions':               
                    self.definition_category_index[key[1]]['bm25_vec'] = bm25_vec
                elif key[0] == 'prompts':
                    self.prompt_category_index[key[1]]['bm25_vec'] = bm25_vec
            elif len(key) == 3:
                # this is an item key
                if key[0] == 'definitions':
                    self.definition_index[key[1]][key[2]]['bm25_vec'] = bm25_vec
                elif key[0] == 'prompts':
                    self.prompt_index[key[1]][key[2]]['bm25_vec'] = bm25_vec

        
        return

    def idf_vectorizer(self):
        self.doc_count = self.doc_term_matrix.shape[0]
        
        doc_freq_vec = torch.count_nonzero(self.doc_term_matrix, dim=0)

        idf_vec = torch.log(((self.doc_count - doc_freq_vec + 0.5) / (doc_freq_vec + 0.5)) + 1)

        return idf_vec
    
    def tf_vectorizer(self):
        doc_lengths = self.doc_term_matrix.sum(dim=1)

        self.avg_length = doc_lengths.mean()

        tf_matrix = (self.doc_term_matrix * (self.k1 + 1)) / (self.doc_term_matrix + self.k1 * (1 - self.beta + self.beta * (doc_lengths / self.avg_length)).unsqueeze(1))

        return tf_matrix
    
    def get_bm25_score(self, query, docs):
        bm25_vec = torch.matmul(docs, query)
        return bm25_vec
    
    def get_cosine_sim(self, query, docs):
        cosine_sim_vec = torch.matmul(docs, query)
        return cosine_sim_vec

    def search(self, query, k1, k2, threshold):
        with torch.no_grad():
            # getting cosine sim
            tokenized_query = self.tokenizer(query, return_tensors='pt', max_length=512, padding='max_length', truncation=True).to(self.device)
            attention_mask = tokenized_query['attention_mask']
            embedded_query = self.model(tokenized_query['input_ids'], attention_mask=attention_mask)['last_hidden_state'].squeeze().to(self.device)
            mask = attention_mask.T.expand(embedded_query.size()).float()
            summed = (embedded_query * mask).sum(dim=1)
            attention_count = torch.clamp(mask.sum(dim=1), min=1e-9)
            normed_query = summed / attention_count

            # getting bm25 score
            query_vec = torch.tensor(self.count_vectorizer.transform([query]).toarray(), dtype=torch.float32).squeeze().to(self.device)
            one_hot_query_vec = torch.clamp(query_vec, 0, 1)

            # searching categories:
            categories = {'definitions': {'index': self.definition_category_index, 'keys': list(self.definition_category_index.keys())}, 'prompts': {'index': self.prompt_category_index, 'keys': list(self.prompt_category_index.keys())}}
            top_categories = {}
            for group in categories:
                bm25_results = self.get_bm25_score(one_hot_query_vec, torch.stack([categories[group]['index'][i]['bm25_vec'] for i in categories[group]['index']]))
                bert_results = self.get_cosine_sim(normed_query, torch.stack([categories[group]['index'][i]['bert_vec'] for i in categories[group]['index']])) * self.bert_weight
                print('bm25 results: ', bm25_results)
                print('bert results: ', bert_results)
                combined_results = bm25_results + bert_results
                top_k_category_indices = torch.topk(combined_results, k1).indices
                top_k_category_keys = [categories[group]['keys'][i] for i in top_k_category_indices]
                top_categories[group] = top_k_category_keys

            print('Top categories: ', top_categories)
            top_items = {}
            items = {'definitions': {'index': self.definition_index, 'keys': []}, 'prompts': {'index': self.prompt_index, 'keys': []}}
            scores = []
            keys = []
            for group in top_categories:
                for item in top_categories[group]:
                    category_keys = [(group, item, i) for i in items[group]['index'][item]]
                    keys.extend(category_keys)
                    bm25_results = self.get_bm25_score(one_hot_query_vec, torch.stack([items[key[0]]['index'][key[1]][key[2]]['bm25_vec'] for key in category_keys])) * self.bm25_weight
                    bert_results = self.get_cosine_sim(normed_query, torch.stack([items[key[0]]['index'][key[1]][key[2]]['bert_vec'] for key in category_keys])) * self.bert_weight
                    combined_results = bm25_results + bert_results
                    scores.extend(combined_results)

            print(scores)
            print(keys)

            if k2 > len(scores):
                k2 = len(scores)
            top_k_items = torch.topk(torch.tensor(scores), k2)
            top_k_item_indices = top_k_items.indices
            top_k_item_scores = top_k_items.values
            print(top_k_item_scores)
            top_k_item_keys = [keys[i] for i in top_k_item_indices]
            for index, key in enumerate(top_k_item_keys):
                if top_k_item_scores[index] >= threshold:
                    if key[0] not in top_items:
                        top_items[key[0]] = {}
                    if key[1] not in top_items[key[0]]:
                        top_items[key[0]][key[1]] = {}
                    top_items[key[0]][key[1]][key[2]] = items[key[0]]['index'][key[1]][key[2]]['item']
            
            print(top_items)
        return top_items

if __name__ == '__main__':
    context = contextManager('contexts', 1.5, .75, 1, 1)
    print('Definition category index: ', context.definition_category_index)
    print('Definition index: ', context.definition_index)
    context.search('closed loans', 1, 3, 40)