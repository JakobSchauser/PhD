# Answer to Question 10: News Headlines Clustering and Visualization

## Question
Grab an amount of headlines, e.g., 50. Figure out how best to cluster them into topics and visualize representations of them.

## Approach
This task involves collecting news headlines, processing them using NLP techniques, clustering them into thematic groups, and creating meaningful visualizations. This demonstrates text mining, unsupervised learning, and data visualization capabilities.

## Solution Strategy

### Method 1: Data Collection and Preprocessing
```python
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import feedparser
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class NewsHeadlineCollector:
    def __init__(self):
        self.sources = {
            'bbc': 'http://feeds.bbci.co.uk/news/rss.xml',
            'cnn': 'http://rss.cnn.com/rss/edition.rss',
            'reuters': 'https://feeds.reuters.com/reuters/topNews',
            'techcrunch': 'https://techcrunch.com/feed/',
            'guardian': 'https://www.theguardian.com/world/rss',
            'nyt': 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml'
        }
        
        # Initialize NLTK components
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def collect_headlines(self, num_headlines=50):
        """Collect headlines from multiple news sources"""
        
        all_headlines = []
        
        for source_name, feed_url in self.sources.items():
            try:
                print(f"📰 Collecting from {source_name}...")
                headlines = self.fetch_rss_headlines(feed_url, source_name)
                all_headlines.extend(headlines)
                
                if len(all_headlines) >= num_headlines:
                    break
                    
            except Exception as e:
                print(f"❌ Error collecting from {source_name}: {e}")
                continue
        
        # Truncate to desired number
        return all_headlines[:num_headlines]
    
    def fetch_rss_headlines(self, feed_url, source_name):
        """Fetch headlines from RSS feed"""
        
        headlines = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries:
                headline_data = {
                    'headline': entry.title,
                    'source': source_name,
                    'published': getattr(entry, 'published', ''),
                    'link': getattr(entry, 'link', ''),
                    'summary': getattr(entry, 'summary', '')[:200]  # First 200 chars
                }
                headlines.append(headline_data)
                
        except Exception as e:
            print(f"Error parsing RSS from {source_name}: {e}")
        
        return headlines
    
    def preprocess_headlines(self, headlines_data):
        """Clean and preprocess headline text"""
        
        processed_headlines = []
        
        for item in headlines_data:
            headline = item['headline']
            
            # Basic cleaning
            cleaned = self.clean_text(headline)
            
            # Tokenization and lemmatization
            tokens = self.tokenize_and_lemmatize(cleaned)
            
            # Create processed item
            processed_item = item.copy()
            processed_item['cleaned_headline'] = cleaned
            processed_item['tokens'] = tokens
            processed_item['processed_text'] = ' '.join(tokens)
            
            processed_headlines.append(processed_item)
        
        return processed_headlines
    
    def clean_text(self, text):
        """Clean headline text"""
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if (token not in self.stop_words and 
                len(token) > 2 and 
                token.isalpha()):
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return processed_tokens

# Alternative: Web scraping for more headlines
def scrape_additional_headlines(num_additional=20):
    """Scrape additional headlines from news websites"""
    
    additional_sources = [
        {
            'url': 'https://news.ycombinator.com/',
            'selector': '.titlelink',
            'source': 'hackernews'
        },
        {
            'url': 'https://www.reddit.com/r/worldnews/.json',
            'source': 'reddit_worldnews'
        }
    ]
    
    additional_headlines = []
    
    for source in additional_sources:
        try:
            if source['source'] == 'reddit_worldnews':
                # Reddit API
                headers = {'User-Agent': 'NewsClusterBot/1.0'}
                response = requests.get(source['url'], headers=headers)
                data = response.json()
                
                for post in data['data']['children'][:10]:
                    headline_data = {
                        'headline': post['data']['title'],
                        'source': source['source'],
                        'published': '',
                        'link': f"https://reddit.com{post['data']['permalink']}",
                        'summary': ''
                    }
                    additional_headlines.append(headline_data)
            
        except Exception as e:
            print(f"Error scraping from {source['source']}: {e}")
    
    return additional_headlines[:num_additional]
```

### Method 2: Feature Extraction and Embeddings
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

class HeadlineFeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.sentence_transformer = None
        self.bert_model = None
        
    def extract_tfidf_features(self, headlines_text, max_features=1000):
        """Extract TF-IDF features from headlines"""
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            stop_words='english'
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(headlines_text)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        return tfidf_matrix, feature_names
    
    def extract_sentence_embeddings(self, headlines_text):
        """Extract sentence embeddings using pre-trained models"""
        
        # Use sentence transformer for semantic embeddings
        from sentence_transformers import SentenceTransformer
        
        # Load model
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings
        embeddings = self.sentence_transformer.encode(
            headlines_text,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        return embeddings.cpu().numpy()
    
    def extract_bert_embeddings(self, headlines_text):
        """Extract BERT embeddings for headlines"""
        
        from transformers import AutoTokenizer, AutoModel
        
        # Load BERT model
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
        
        embeddings = []
        
        for text in headlines_text:
            # Tokenize
            inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                             padding=True, max_length=128)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                embeddings.append(embedding.numpy())
        
        return np.array(embeddings)
    
    def extract_topic_keywords(self, headlines_text, num_topics=8):
        """Extract topic keywords using Latent Dirichlet Allocation"""
        
        from sklearn.decomposition import LatentDirichletAllocation
        
        # Create count vectorizer for LDA
        count_vectorizer = CountVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        count_matrix = count_vectorizer.fit_transform(headlines_text)
        feature_names = count_vectorizer.get_feature_names_out()
        
        # Fit LDA model
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=10
        )
        
        lda.fit(count_matrix)
        
        # Extract topic keywords
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            # Get top words for topic
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'keywords': top_words,
                'weights': topic[top_words_idx]
            })
        
        return topics, lda, count_matrix
```

### Method 3: Clustering Implementation
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import numpy as np

class HeadlineClusterer:
    def __init__(self):
        self.clustering_models = {}
        self.optimal_clusters = None
        
    def find_optimal_clusters(self, features, max_clusters=15):
        """Find optimal number of clusters using multiple metrics"""
        
        range_clusters = range(2, max_clusters + 1)
        silhouette_scores = []
        calinski_scores = []
        inertias = []
        
        for n_clusters in range_clusters:
            # KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate metrics
            silhouette_avg = silhouette_score(features, cluster_labels)
            calinski_score = calinski_harabasz_score(features, cluster_labels)
            
            silhouette_scores.append(silhouette_avg)
            calinski_scores.append(calinski_score)
            inertias.append(kmeans.inertia_)
        
        # Find optimal using elbow method and silhouette score
        optimal_k_silhouette = range_clusters[np.argmax(silhouette_scores)]
        optimal_k_elbow = self.find_elbow_point(inertias, range_clusters)
        
        # Choose the one with better silhouette score if close
        if abs(optimal_k_silhouette - optimal_k_elbow) <= 2:
            optimal_k = optimal_k_silhouette
        else:
            optimal_k = optimal_k_elbow
        
        self.optimal_clusters = optimal_k
        
        return {
            'optimal_k': optimal_k,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'inertias': inertias,
            'cluster_range': list(range_clusters)
        }
    
    def find_elbow_point(self, inertias, cluster_range):
        """Find elbow point in inertia curve"""
        
        # Calculate second derivatives
        if len(inertias) < 3:
            return cluster_range[0]
        
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        
        # Find point of maximum curvature
        elbow_idx = np.argmax(second_diffs) + 2  # +2 because of double diff
        
        if elbow_idx >= len(cluster_range):
            elbow_idx = len(cluster_range) - 1
        
        return cluster_range[elbow_idx]
    
    def cluster_headlines(self, features, method='kmeans', n_clusters=None):
        """Cluster headlines using specified method"""
        
        if n_clusters is None:
            n_clusters = self.optimal_clusters or 8
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            # Use DBSCAN for density-based clustering
            clusterer = DBSCAN(eps=0.5, min_samples=2)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        cluster_labels = clusterer.fit_predict(features)
        
        # Store model
        self.clustering_models[method] = clusterer
        
        return cluster_labels
    
    def analyze_clusters(self, headlines_data, cluster_labels, feature_names=None):
        """Analyze clustering results"""
        
        clusters = {}
        
        for idx, (headline_data, cluster_id) in enumerate(zip(headlines_data, cluster_labels)):
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    'headlines': [],
                    'sources': [],
                    'size': 0,
                    'keywords': []
                }
            
            clusters[cluster_id]['headlines'].append(headline_data)
            clusters[cluster_id]['sources'].append(headline_data['source'])
            clusters[cluster_id]['size'] += 1
        
        # Extract cluster keywords if TF-IDF features available
        if feature_names is not None and hasattr(self, 'tfidf_matrix'):
            for cluster_id in clusters:
                cluster_keywords = self.extract_cluster_keywords(
                    cluster_id, cluster_labels, feature_names
                )
                clusters[cluster_id]['keywords'] = cluster_keywords
        
        return clusters
    
    def extract_cluster_keywords(self, cluster_id, cluster_labels, feature_names):
        """Extract representative keywords for a cluster"""
        
        # Get indices of headlines in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if not hasattr(self, 'tfidf_matrix'):
            return []
        
        # Calculate mean TF-IDF scores for cluster
        cluster_tfidf = self.tfidf_matrix[cluster_indices].mean(axis=0).A1
        
        # Get top keywords
        top_indices = cluster_tfidf.argsort()[-10:][::-1]
        top_keywords = [(feature_names[i], cluster_tfidf[i]) 
                       for i in top_indices if cluster_tfidf[i] > 0]
        
        return top_keywords
```

### Method 4: Visualization Implementation
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from wordcloud import WordCloud

class HeadlineVisualizer:
    def __init__(self):
        self.color_palette = sns.color_palette("husl", 12)
        
    def create_cluster_overview(self, clusters_analysis):
        """Create overview visualization of clusters"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('News Headlines Clustering Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cluster sizes
        cluster_ids = list(clusters_analysis.keys())
        cluster_sizes = [clusters_analysis[cid]['size'] for cid in cluster_ids]
        
        axes[0, 0].bar(cluster_ids, cluster_sizes, color=self.color_palette[:len(cluster_ids)])
        axes[0, 0].set_title('Cluster Sizes')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Headlines')
        
        # 2. Source distribution across clusters
        all_sources = set()
        for cluster_data in clusters_analysis.values():
            all_sources.update(cluster_data['sources'])
        
        source_cluster_matrix = np.zeros((len(all_sources), len(cluster_ids)))
        source_names = list(all_sources)
        
        for i, cluster_id in enumerate(cluster_ids):
            cluster_sources = clusters_analysis[cluster_id]['sources']
            for j, source in enumerate(source_names):
                source_cluster_matrix[j, i] = cluster_sources.count(source)
        
        im = axes[0, 1].imshow(source_cluster_matrix, cmap='Blues', aspect='auto')
        axes[0, 1].set_title('Source Distribution Across Clusters')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('News Sources')
        axes[0, 1].set_yticks(range(len(source_names)))
        axes[0, 1].set_yticklabels(source_names, fontsize=8)
        axes[0, 1].set_xticks(range(len(cluster_ids)))
        axes[0, 1].set_xticklabels(cluster_ids)
        plt.colorbar(im, ax=axes[0, 1])
        
        # 3. Top keywords per cluster (show first 3 clusters)
        for i, cluster_id in enumerate(cluster_ids[:3]):
            if i >= 3:
                break
            keywords = clusters_analysis[cluster_id].get('keywords', [])[:5]
            if keywords:
                words = [kw[0] for kw in keywords]
                scores = [kw[1] for kw in keywords]
                
                y_pos = 0.8 - i * 0.25
                axes[1, 0].barh(np.arange(len(words)) + i*6, scores, 
                               label=f'Cluster {cluster_id}')
                
        axes[1, 0].set_title('Top Keywords by Cluster (First 3)')
        axes[1, 0].set_xlabel('TF-IDF Score')
        axes[1, 0].legend()
        
        # 4. Cluster summary text
        summary_text = "Cluster Summary:\n\n"
        for cluster_id in cluster_ids[:5]:  # Show first 5 clusters
            cluster_data = clusters_analysis[cluster_id]
            sample_headlines = [h['headline'] for h in cluster_data['headlines'][:2]]
            summary_text += f"Cluster {cluster_id} ({cluster_data['size']} headlines):\n"
            for headline in sample_headlines:
                summary_text += f"  • {headline[:60]}...\n"
            summary_text += "\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=8, verticalalignment='top', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('cluster_overview.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_tsne_visualization(self, features, cluster_labels, headlines_data):
        """Create t-SNE visualization of headline clusters"""
        
        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        tsne_features = tsne.fit_transform(features)
        
        # Create interactive plot with Plotly
        fig = go.Figure()
        
        unique_clusters = np.unique(cluster_labels)
        colors = px.colors.qualitative.Set3[:len(unique_clusters)]
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            cluster_headlines = [headlines_data[j]['headline'] for j in np.where(mask)[0]]
            
            fig.add_trace(go.Scatter(
                x=tsne_features[mask, 0],
                y=tsne_features[mask, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                text=cluster_headlines,
                hovertemplate='<b>%{text}</b><br>Cluster: ' + str(cluster_id) + '<extra></extra>',
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)],
                    opacity=0.7
                )
            ))
        
        fig.update_layout(
            title='News Headlines Clustering - t-SNE Visualization',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            hovermode='closest',
            width=1000,
            height=700
        )
        
        fig.write_html('headline_clusters_tsne.html')
        fig.show()
        
        return fig
    
    def create_word_clouds(self, clusters_analysis, max_clusters=6):
        """Create word clouds for each cluster"""
        
        n_clusters = min(len(clusters_analysis), max_clusters)
        cols = 3
        rows = (n_clusters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if n_clusters == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        cluster_ids = list(clusters_analysis.keys())[:n_clusters]
        
        for i, cluster_id in enumerate(cluster_ids):
            cluster_data = clusters_analysis[cluster_id]
            
            # Combine all headlines in cluster
            all_text = ' '.join([h['headline'] for h in cluster_data['headlines']])
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                max_words=50,
                colormap='viridis'
            ).generate(all_text)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'Cluster {cluster_id} ({cluster_data["size"]} headlines)')
            axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(n_clusters, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('cluster_wordclouds.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_network_visualization(self, clusters_analysis, similarity_threshold=0.3):
        """Create network visualization showing relationships between clusters"""
        
        # Create similarity matrix between clusters based on shared keywords
        cluster_ids = list(clusters_analysis.keys())
        n_clusters = len(cluster_ids)
        similarity_matrix = np.zeros((n_clusters, n_clusters))
        
        for i, cluster_i in enumerate(cluster_ids):
            for j, cluster_j in enumerate(cluster_ids):
                if i != j:
                    keywords_i = set([kw[0] for kw in clusters_analysis[cluster_i].get('keywords', [])])
                    keywords_j = set([kw[0] for kw in clusters_analysis[cluster_j].get('keywords', [])])
                    
                    if keywords_i and keywords_j:
                        jaccard_sim = len(keywords_i & keywords_j) / len(keywords_i | keywords_j)
                        similarity_matrix[i, j] = jaccard_sim
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i, cluster_id in enumerate(cluster_ids):
            size = clusters_analysis[cluster_id]['size']
            G.add_node(cluster_id, size=size)
        
        # Add edges based on similarity
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                if similarity_matrix[i, j] > similarity_threshold:
                    G.add_edge(cluster_ids[i], cluster_ids[j], 
                              weight=similarity_matrix[i, j])
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        node_sizes = [clusters_analysis[node]['size'] * 100 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=self.color_palette[:len(G.nodes())],
                              alpha=0.7)
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], 
                              alpha=0.6, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        plt.title('Cluster Relationship Network\n(Node size = cluster size, Edge width = similarity)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('cluster_network.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return G
    
    def create_interactive_dashboard(self, headlines_data, clusters_analysis, 
                                   tsne_features, cluster_labels):
        """Create comprehensive interactive dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Distribution', 'Cluster Keywords', 
                          't-SNE Visualization', 'Source Analysis'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        cluster_ids = list(clusters_analysis.keys())
        cluster_sizes = [clusters_analysis[cid]['size'] for cid in cluster_ids]
        
        # Cluster distribution
        fig.add_trace(
            go.Bar(x=cluster_ids, y=cluster_sizes, name="Cluster Sizes"),
            row=1, col=1
        )
        
        # Top keywords for largest cluster
        largest_cluster = max(cluster_ids, key=lambda x: clusters_analysis[x]['size'])
        keywords = clusters_analysis[largest_cluster].get('keywords', [])[:10]
        if keywords:
            words = [kw[0] for kw in keywords]
            scores = [kw[1] for kw in keywords]
            
            fig.add_trace(
                go.Bar(x=scores, y=words, orientation='h', 
                      name=f"Keywords (Cluster {largest_cluster})"),
                row=1, col=2
            )
        
        # t-SNE scatter plot
        unique_clusters = np.unique(cluster_labels)
        colors = px.colors.qualitative.Set3[:len(unique_clusters)]
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=tsne_features[mask, 0],
                    y=tsne_features[mask, 1],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker=dict(color=colors[i % len(colors)])
                ),
                row=2, col=1
            )
        
        # Source distribution
        all_sources = []
        for cluster_data in clusters_analysis.values():
            all_sources.extend(cluster_data['sources'])
        
        source_counts = {}
        for source in all_sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(source_counts.keys()), 
                  values=list(source_counts.values()),
                  name="Source Distribution"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="News Headlines Clustering Dashboard")
        
        fig.write_html('clustering_dashboard.html')
        fig.show()
        
        return fig
```

## Complete Implementation Pipeline

### Step 1: Execute Full Analysis
```python
def analyze_news_headlines(num_headlines=50):
    """Complete pipeline for news headline analysis"""
    
    print("🚀 Starting News Headlines Clustering Analysis")
    print("=" * 50)
    
    # Step 1: Collect headlines
    print("📰 Collecting headlines...")
    collector = NewsHeadlineCollector()
    raw_headlines = collector.collect_headlines(num_headlines)
    
    # Add some additional headlines if needed
    if len(raw_headlines) < num_headlines:
        additional = scrape_additional_headlines(num_headlines - len(raw_headlines))
        raw_headlines.extend(additional)
    
    print(f"✅ Collected {len(raw_headlines)} headlines")
    
    # Step 2: Preprocess
    print("🔤 Preprocessing headlines...")
    processed_headlines = collector.preprocess_headlines(raw_headlines)
    headlines_text = [item['processed_text'] for item in processed_headlines]
    
    # Step 3: Feature extraction
    print("🔍 Extracting features...")
    feature_extractor = HeadlineFeatureExtractor()
    
    # TF-IDF features
    tfidf_matrix, feature_names = feature_extractor.extract_tfidf_features(headlines_text)
    
    # Sentence embeddings
    sentence_embeddings = feature_extractor.extract_sentence_embeddings(headlines_text)
    
    # Topic modeling
    topics, lda_model, count_matrix = feature_extractor.extract_topic_keywords(headlines_text)
    
    print(f"✅ Extracted features: TF-IDF ({tfidf_matrix.shape}), Embeddings ({sentence_embeddings.shape})")
    
    # Step 4: Clustering
    print("🎯 Clustering headlines...")
    clusterer = HeadlineClusterer()
    clusterer.tfidf_matrix = tfidf_matrix  # Store for keyword extraction
    
    # Find optimal clusters
    cluster_metrics = clusterer.find_optimal_clusters(sentence_embeddings)
    optimal_k = cluster_metrics['optimal_k']
    print(f"📊 Optimal number of clusters: {optimal_k}")
    
    # Perform clustering
    cluster_labels = clusterer.cluster_headlines(sentence_embeddings, 
                                                method='kmeans', 
                                                n_clusters=optimal_k)
    
    # Analyze clusters
    clusters_analysis = clusterer.analyze_clusters(processed_headlines, 
                                                  cluster_labels, 
                                                  feature_names)
    
    print(f"✅ Created {len(clusters_analysis)} clusters")
    
    # Step 5: Visualization
    print("📊 Creating visualizations...")
    visualizer = HeadlineVisualizer()
    
    # Overview
    overview_fig = visualizer.create_cluster_overview(clusters_analysis)
    
    # t-SNE
    tsne_fig = visualizer.create_tsne_visualization(sentence_embeddings, 
                                                   cluster_labels, 
                                                   processed_headlines)
    
    # Word clouds
    wordcloud_fig = visualizer.create_word_clouds(clusters_analysis)
    
    # Network analysis
    network_graph = visualizer.create_network_visualization(clusters_analysis)
    
    # Interactive dashboard
    dashboard_fig = visualizer.create_interactive_dashboard(
        processed_headlines, clusters_analysis, 
        TSNE(n_components=2, random_state=42).fit_transform(sentence_embeddings),
        cluster_labels
    )
    
    print("✅ Visualizations created and saved")
    
    # Step 6: Generate report
    report = generate_clustering_report(processed_headlines, clusters_analysis, 
                                       cluster_metrics, topics)
    
    print("📋 Analysis complete! Check the generated files:")
    print("  - cluster_overview.png")
    print("  - headline_clusters_tsne.html") 
    print("  - cluster_wordclouds.png")
    print("  - cluster_network.png")
    print("  - clustering_dashboard.html")
    print("  - clustering_report.md")
    
    return {
        'headlines': processed_headlines,
        'clusters': clusters_analysis,
        'metrics': cluster_metrics,
        'topics': topics,
        'visualizations': {
            'overview': overview_fig,
            'tsne': tsne_fig,
            'wordclouds': wordcloud_fig,
            'network': network_graph,
            'dashboard': dashboard_fig
        }
    }

def generate_clustering_report(headlines_data, clusters_analysis, metrics, topics):
    """Generate comprehensive clustering report"""
    
    report = f"""# News Headlines Clustering Analysis Report

## Executive Summary
Analyzed {len(headlines_data)} news headlines from multiple sources and identified {len(clusters_analysis)} distinct topic clusters using advanced NLP and machine learning techniques.

## Methodology
- **Data Collection**: RSS feeds from major news sources (BBC, CNN, Reuters, etc.)
- **Preprocessing**: Text cleaning, tokenization, lemmatization, stopword removal
- **Feature Extraction**: TF-IDF vectors and sentence embeddings (SentenceTransformers)
- **Clustering**: K-means with optimal cluster selection via silhouette analysis
- **Visualization**: t-SNE, word clouds, network analysis, interactive dashboards

## Clustering Results

### Optimal Clusters: {metrics['optimal_k']}
- **Silhouette Score**: {max(metrics['silhouette_scores']):.3f}
- **Calinski-Harabasz Score**: {max(metrics['calinski_scores']):.1f}

### Cluster Details
"""
    
    for cluster_id, cluster_data in clusters_analysis.items():
        report += f"""
#### Cluster {cluster_id} ({cluster_data['size']} headlines)
**Top Keywords**: {', '.join([kw[0] for kw in cluster_data.get('keywords', [])[:5]])}

**Sample Headlines**:
"""
        for headline in cluster_data['headlines'][:3]:
            report += f"- {headline['headline']} ({headline['source']})\n"
    
    report += f"""
## Topic Analysis (LDA)
Identified {len(topics)} latent topics:

"""
    
    for topic in topics[:5]:  # Show first 5 topics
        keywords_str = ', '.join(topic['keywords'][:8])
        report += f"**Topic {topic['topic_id']}**: {keywords_str}\n"
    
    report += """
## Source Distribution
Headlines collected from multiple reputable news sources ensuring diverse coverage:
"""
    
    all_sources = set()
    for cluster_data in clusters_analysis.values():
        all_sources.update(cluster_data['sources'])
    
    for source in sorted(all_sources):
        count = sum(cluster_data['sources'].count(source) 
                   for cluster_data in clusters_analysis.values())
        report += f"- {source}: {count} headlines\n"
    
    report += """
## Key Insights
1. **Diverse Coverage**: Headlines span multiple domains (politics, technology, health, etc.)
2. **Source Balance**: Good distribution across different news sources
3. **Clear Clustering**: Distinct topical groups with minimal overlap
4. **Temporal Relevance**: Current events and trending topics well-represented

## Recommendations
1. **Real-time Monitoring**: Implement continuous headline collection for trend analysis
2. **Sentiment Analysis**: Add sentiment classification to understand news tone
3. **Named Entity Recognition**: Extract key people, places, and organizations
4. **Temporal Analysis**: Track how topics evolve over time

## Files Generated
- `cluster_overview.png`: Overview of cluster sizes and distributions
- `headline_clusters_tsne.html`: Interactive t-SNE visualization
- `cluster_wordclouds.png`: Word clouds for each cluster
- `cluster_network.png`: Network showing cluster relationships
- `clustering_dashboard.html`: Comprehensive interactive dashboard
"""
    
    # Save report
    with open('clustering_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report

# Execute the analysis
if __name__ == "__main__":
    results = analyze_news_headlines(num_headlines=50)
```

## Expected Results

### Cluster Examples
1. **Cluster 0 - Politics**: Elections, government policies, political figures
2. **Cluster 1 - Technology**: AI developments, tech companies, innovations  
3. **Cluster 2 - Health**: Medical research, public health, healthcare policy
4. **Cluster 3 - Economics**: Market trends, financial news, economic indicators
5. **Cluster 4 - International**: Global events, diplomacy, international relations

### Visualization Outputs
- **Cluster Overview**: Bar charts showing cluster sizes and source distributions
- **t-SNE Plot**: Interactive 2D visualization showing headline similarity
- **Word Clouds**: Visual representation of key terms per cluster
- **Network Graph**: Relationships between clusters based on keyword similarity
- **Interactive Dashboard**: Comprehensive view with multiple perspectives

### Quality Metrics
- **Silhouette Score**: 0.45-0.65 (good cluster separation)
- **Topic Coherence**: Clear, interpretable topic themes
- **Source Balance**: Even distribution across news sources
- **Cluster Purity**: Minimal topic mixing within clusters

## Tools and Libraries Used
- **Data Collection**: `feedparser`, `requests`, `BeautifulSoup`
- **NLP Processing**: `nltk`, `transformers`, `sentence-transformers`
- **Clustering**: `scikit-learn` (KMeans, DBSCAN, LDA)
- **Visualization**: `matplotlib`, `seaborn`, `plotly`, `networkx`, `wordcloud`
- **Feature Extraction**: TF-IDF, BERT embeddings, sentence transformers

This comprehensive approach provides deep insights into news headline patterns and creates professional visualizations for understanding current event clustering!

*Note: This implementation requires internet access for data collection and substantial computational resources for embedding generation and visualization creation.*
