# 🎫 Customer Support Ticket Routing System

An AI-powered ticket classification system built using **FastAPI + OpenEnv + LLM (HuggingFace/OpenAI compatible)** that automatically routes customer support tickets into the correct category.

---

## 🚀 Project Overview

This project simulates a **customer support environment** where an AI agent classifies incoming tickets into one of four categories:

- 💳 Billing  
- 🛠️ Technical  
- 📄 General  
- 🚨 Urgent  

The system uses:
- Rule-based fallback logic  
- LLM-based classification  
- Reinforcement-style environment scoring  

---

## 🧠 Features

- ✅ AI-based ticket classification  
- ✅ OpenEnv compliant environment  
- ✅ FastAPI backend with API endpoints  
- ✅ Reward-based evaluation system  
- ✅ Rule-based fallback if LLM fails  
- ✅ Multiple difficulty levels (easy, medium, hard)  
- ✅ Real-time simulation results  

---

## 🏗️ Tech Stack

- **Backend:** FastAPI  
- **AI/LLM:** HuggingFace / OpenAI compatible API  
- **Environment:** OpenEnv Framework  
- **Language:** Python  
- **Deployment Ready:** Docker  

---

## 📂 Project Structure
├── app.py # FastAPI server & API routes
├── env.py # Ticket routing environment logic
├── inference.py # LLM + agent execution
├── tasks.py # Ticket datasets (easy/medium/hard)
├── requirements.txt # Dependencies
├── openenv.yaml # OpenEnv configuration
├── Dockerfile # Container setup


---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone <your-repo-link>
cd <project-folder>

2️⃣ Install dependencies
pip install -r requirements.txt

## ▶️ Run the Project

### Start FastAPI Server

Run the following command to start the backend server:

```bash
uvicorn app:app --reload
