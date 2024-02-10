# Pytorch day 1
I started my journey in learning pytorch, so here are some key points which I understood from the 1hr of the video from freeCodeCamp.org (https://youtu.be/V_xro1bcAuA?si=stuE-gxwsMDXe4nj):
# Deep Learning: 
Deep learning is a subset of machine learning that utilizes artificial neural networks with many layers (deep architectures) to model and understand complex patterns in data. It aims to mimic the human brain's structure and function by using interconnected layers of nodes (neurons) that process information. Deep learning has shown remarkable success in various tasks such as image and speech recognition, natural language processing, and reinforcement learning.

# Machine Learning:
Machine learning is a broader field that focuses on developing algorithms and techniques that allow computers to learn from and make predictions or decisions based on data without being explicitly programmed. It involves the study of statistical models and algorithms that enable systems to perform specific tasks without being explicitly programmed for those tasks. Machine learning encompasses various approaches, including supervised learning, unsupervised learning, semi-supervised learning, reinforcement learning, and more.

# Neural Networks:
Neural networks are computing systems inspired by the biological neural networks of animal brains. They consist of interconnected nodes (neurons) organized in layers. Each neuron receives input, processes it through an activation function, and produces an output. Neural networks can learn complex patterns and relationships in data by adjusting the weights associated with connections between neurons during a training process.

# Difference between Deep Learning and Machine Learning:
The primary difference between deep learning and traditional machine learning lies in the complexity of the models used and the nature of the features they learn. Deep learning models, particularly deep neural networks, have many layers and can automatically learn hierarchical representations of data, often without requiring handcrafted features. Traditional machine learning algorithms, on the other hand, typically rely on feature engineering, where domain experts manually select and engineer relevant features from the input data.

# Learning Paradigms:
Learning paradigms refer to the different approaches or methodologies used in machine learning and deep learning. Some common learning paradigms include:

Supervised Learning: Learning from labeled data, where the algorithm learns to map input data to known output labels.
Unsupervised Learning: Learning from unlabeled data, where the algorithm discovers patterns and structures in the input data without explicit supervision.
Reinforcement Learning: Learning through trial and error by interacting with an environment to maximize rewards or achieve predefined goals.
Semi-Supervised Learning: Learning from a combination of labeled and unlabeled data to improve performance, particularly when labeled data is scarce.
Self-Supervised Learning: Learning from the data itself without explicit labels, often by generating auxiliary tasks from the data.

# PyTorch: 
PyTorch is an open-source machine learning framework developed by Facebook's AI Research lab. It provides a flexible and dynamic computational graph that allows developers to define and execute computational graphs on-the-fly. PyTorch is widely used for various deep learning tasks, including neural network training, model deployment, and research prototyping. It supports dynamic computation graphs, making it easier to debug and experiment with models compared to static graph frameworks. PyTorch also offers extensive support for GPU acceleration, enabling efficient training of large-scale models.

# Tensors: 
Tensors are the fundamental data structures used in PyTorch and other deep learning frameworks. They are multi-dimensional arrays similar to NumPy arrays but optimized for GPU acceleration and deep learning operations. Tensors can have different numbers of dimensions, including scalars (0-dimensional tensors), vectors (1-dimensional tensors), matrices (2-dimensional tensors), and higher-dimensional tensors. Tensors encapsulate both the data and the operations that can be performed on that data, allowing for efficient computation and manipulation of data in deep learning models.
