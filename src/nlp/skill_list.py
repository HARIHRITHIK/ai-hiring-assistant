# src/nlp/skill_list.py
"""A comprehensive, curated taxonomy of technical skills and domain mappings for ATS analysis."""

SKILL_SET = {
    # AI / Machine Learning / Deep Learning / LLMs
    "ai",
    "artificial intelligence",
    "ai engineer",
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "llama",
    "llama2",
    "llama3",
    "qlora",
    "lora",
    "peft",
    "generative ai",
    "genai",
    "transformers",
    "huggingface",
    "fine-tuning",
    "prompt engineering",
    "rag",
    "retrieval augmented generation",
    "vector database",
    "chromadb",
    "pinecone",
    "weaviate",
    "qdrant",
    "faiss",
    "neural networks",
    "cnn",
    "rnn",
    "lstm",
    "computer vision",
    "object detection",
    "scikit-learn",
    "pytorch",
    "tensorflow",
    "keras",
    "opencv",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "plotly",
    "data science",
    "data analysis",
    "data engineering",
    "data visualization",
    "big data",
    "spark",
    "pyspark",
    "hadoop",
    "etl",
    "statistics",
    "probability",
    "regression",
    "classification",
    "clustering",
    "reinforcement learning",
    "langchain",
    "llamaindex",
    "crewai",
    "autogen",
    "ollama",
    "vllm",
    "streamlit",
    "gradio",
    "mlops",
    "mlflow",
    "wandb",
    "dvc",

    # Core Programming Languages
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "sql",
    "nosql",
    "mongodb",
    "postgresql",
    "mysql",
    "sqlite",
    "redis",
    "go",
    "golang",
    "rust",
    "kotlin",
    "swift",
    "r",
    "scala",
    "ruby",
    "php",
    "dart",
    "matlab",
    "bash",
    "shell scripting",
    "powershell",

    # Cloud Platforms & Infrastructure
    "aws",
    "azure",
    "gcp",
    "google cloud",
    "amazon web services",
    "cloud computing",
    "cloud services",
    "ec2",
    "s3",
    "lambda",
    "docker",
    "kubernetes",
    "k8s",
    "terraform",
    "ansible",
    "jenkins",
    "git",
    "github",
    "gitlab",
    "bitbucket",
    "linux",
    "unix",
    "ci/cd",
    "devops",
    "microservices",
    "serverless",
    "rest api",
    "graphql",
    "grpc",
    "nginx",
    "apache",
    "kafka",
    "rabbitmq",
    "elasticsearch",

    # Backend & Web Frameworks
    "fastapi",
    "flask",
    "django",
    "spring",
    "spring boot",
    "nodejs",
    "express",
    "nest.js",
    "ruby on rails",
    "asp.net",
    "react",
    "angular",
    "vue",
    "nextjs",
    "html",
    "html5",
    "css",
    "css3",
    "sass",
    "bootstrap",
    "tailwindcss",
    "flutter",
    "react native",
    "svelte",

    # Computer Science & Software Engineering Fundamentals
    "algorithm",
    "algorithms",
    "data structures",
    "software engineering",
    "full stack",
    "frontend",
    "backend",
    "system design",
    "software architecture",
    "design patterns",
    "oop",
    "object oriented programming",
    "agile",
    "scrum",
    "jira",
    "testing",
    "unit testing",
    "pytest",
    "unittest",
    "automation",
    "selenium",
    "web scraping",
    "beautifulsoup",
    "scrapy",
    "api development",
    "pydantic",

    # Data Warehousing & Analytics
    "snowflake",
    "databricks",
    "bigquery",
    "redshift",
    "dbt",
    "tableau",
    "powerbi",
    "excel",

    # Cybersecurity & Networking
    "cybersecurity",
    "penetration testing",
    "network security",
    "encryption",
    "cryptography",
    "firewall",
    "oauth",
    "jwt",

    # Mobile
    "android",
    "ios",
    "mobile development",
}

# Domain canonical skill mapping to handle short role titles
DOMAIN_SKILL_EXPANSION = {
    "ai engineer": {
        "python", "machine learning", "deep learning", "nlp", "transformers",
        "pytorch", "tensorflow", "algorithms", "neural networks", "data science", "rag"
    },
    "ai": {
        "python", "machine learning", "deep learning", "nlp", "data science"
    },
    "machine learning engineer": {
        "python", "machine learning", "deep learning", "scikit-learn",
        "pytorch", "tensorflow", "data science", "pandas", "numpy", "algorithms", "mlops"
    },
    "ml engineer": {
        "python", "machine learning", "deep learning", "scikit-learn",
        "pytorch", "tensorflow", "data science", "pandas", "numpy"
    },
    "data scientist": {
        "python", "data science", "data analysis", "machine learning",
        "pandas", "numpy", "sql", "statistics", "data visualization", "scikit-learn"
    },
    "data analyst": {
        "python", "sql", "data analysis", "pandas", "data visualization", "statistics", "excel", "powerbi"
    },
    "data engineer": {
        "python", "sql", "etl", "data engineering", "spark", "hadoop", "big data", "aws", "data warehousing"
    },
    "python developer": {
        "python", "django", "flask", "fastapi", "sql", "git", "docker", "rest api", "pytest"
    },
    "full stack developer": {
        "javascript", "python", "react", "html", "css", "nodejs", "sql", "git", "rest api", "docker"
    },
    "full stack engineer": {
        "javascript", "python", "react", "html", "css", "nodejs", "sql", "git", "rest api", "docker"
    },
    "frontend developer": {
        "javascript", "react", "html", "css", "typescript", "git", "tailwindcss"
    },
    "frontend engineer": {
        "javascript", "react", "html", "css", "typescript", "git", "tailwindcss"
    },
    "react developer": {
        "react", "javascript", "typescript", "html", "css", "nodejs", "git", "rest api"
    },
    "backend developer": {
        "python", "java", "sql", "postgresql", "mongodb", "docker", "microservices", "rest api", "git"
    },
    "backend engineer": {
        "python", "java", "sql", "postgresql", "mongodb", "docker", "microservices", "rest api", "git"
    },
    "web developer": {
        "javascript", "html", "css", "react", "nodejs", "sql", "git", "rest api"
    },
    "software engineer": {
        "python", "java", "algorithms", "data structures", "git", "sql", "system design", "oop"
    },
    "software developer": {
        "python", "java", "algorithms", "data structures", "git", "sql", "oop"
    },
    "devops engineer": {
        "docker", "kubernetes", "terraform", "ci/cd", "aws", "linux", "bash", "git", "jenkins", "ansible"
    },
    "devops": {
        "docker", "kubernetes", "terraform", "ci/cd", "aws", "linux", "bash", "git"
    },
    "cloud engineer": {
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "linux", "ci/cd"
    },
    "cloud architect": {
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "system design", "microservices"
    },
    "mobile developer": {
        "android", "ios", "flutter", "react native", "dart", "kotlin", "swift", "git"
    },
    "android developer": {
        "android", "kotlin", "java", "git", "rest api", "sql"
    },
    "ios developer": {
        "ios", "swift", "git", "rest api"
    },
    "java developer": {
        "java", "spring", "spring boot", "sql", "git", "rest api", "docker", "microservices"
    },
    "nlp engineer": {
        "python", "nlp", "transformers", "pytorch", "tensorflow", "deep learning", "machine learning", "rag"
    },
    "deep learning engineer": {
        "python", "deep learning", "pytorch", "tensorflow", "neural networks", "computer vision", "nlp"
    },
    "generative ai engineer": {
        "python", "generative ai", "llama2", "transformers", "pytorch", "rag", "langchain", "prompt engineering"
    },
    "cybersecurity analyst": {
        "cybersecurity", "penetration testing", "network security", "linux", "python", "encryption"
    },
}

# Job-title words to filter from "missing skills" — these are role labels, not actual skills
JOB_TITLE_FILTER_WORDS = {
    "engineer", "developer", "analyst", "architect", "specialist",
    "manager", "lead", "senior", "junior", "intern", "associate",
    "consultant", "administrator", "designer", "officer", "executive",
}
