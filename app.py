import torch
import torch.nn as nn
import streamlit as st
import re

def process_input_user(text, vocab, max_length):
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    encoded_tokens = [vocab.get(word, vocab.get('<UNK>',1)) for word in tokens]
    if len(encoded_tokens) > max_length:
        padding = encoded_tokens[:max_length]
    else:
        padding = encoded_tokens + [0] * (max_length - len(encoded_tokens))
    return torch.tensor([padding], dtype=torch.long)
    

class ChatbotNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.network = nn.Sequential(
            nn.Linear(max_length * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        return self.network(x)

@st.cache_resource
def load_chatbot():
    checkpoint = torch.load("chatbot_model.pth", map_location="cpu", weights_only=False)
    vocab = checkpoint["vocab"]
    labels_to_id = checkpoint["labels_to_id"]
    id_to_label = checkpoint["id_to_label"]
    embedding_dim = checkpoint["embedding_dim"]
    num_classes = checkpoint["num_classes"]
    vocab_size = len(vocab)
    max_length = 17
    model = ChatbotNN(vocab_size, embedding_dim, max_length, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, vocab, id_to_label, max_length
model, vocab, id_to_label, max_length = load_chatbot()
st.set_page_config(page_title="Follow Soulity31 on Github", page_icon="🤖")
st.title("Intent Classification Bot")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type content here"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role":"user", "content":user_input})
    input_tensor = process_input_user(user_input, vocab, max_length)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction_id = torch.max(probabilities, dim=1)
    intent = id_to_label[prediction_id.item()]
    confidence_score = confidence.item() * 100
    if confidence_score <=75:
        bot_response = "I don't know ask gpt"
    else:
        bot_response = f"Predicted Intent: **{intent}** *(Confidence: {confidence_score:2f})*%"
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.messages.append({"role":"assistant","content":bot_response})
    
    