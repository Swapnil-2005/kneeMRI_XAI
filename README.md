# 🧠 NeuroRad Vision  
## Explainable AI-Based Knee Ligament Tear Detection

NeuroRad Vision is an AI-assisted clinical decision support system designed for automated detection and explainable analysis of knee ligament injuries from MRI scans. The system focuses on assisting radiologists and orthopedic specialists through deep learning-based localization, classification, explainability, and multilingual report generation.

---

# 🚀 Features

- ✅ YOLO-based ligament localization using bounding box detection
- ✅ Classification of:
  - ACL Tear
  - ACL Normal
  - PCL Tear
  - PCL Normal
- ✅ Vision Transformer (ViT)-based deep learning classification
- ✅ Grad-CAM based Explainable AI (XAI)
- ✅ Automated AI-generated clinical reports
- ✅ Multilingual report generation using SARVAM API
- ✅ Doctor-centric workflow and clinical UI
- ✅ Modern web interface for MRI analysis

---

# 🖼️ Project Preview

## MRI Detection
![Detection](\assets\detection.jpeg)

## Report Gen
![GradCAM](assets\report.jpeg)



---

# 🏗️ System Architecture

1. MRI Image Upload
2. YOLO-based Ligament Localization
3. ViT-based Classification
4. Grad-CAM Explainability
5. Automated Clinical Report Generation
6. Indic Language Translation using SARVAM API

---

# 🧠 Tech Stack

## Deep Learning
- YOLOv12
- Vision Transformer (ViT)
- TensorFlow
- PyTorch

## Explainability
- Grad-CAM

## Backend / UI
- Streamlit
- FastAPI

## Other Tools
- OpenCV
- NumPy
- Matplotlib
- SARVAM API

---

# 📊 Dataset

The dataset used for this project consists of publicly available knee MRI images containing ACL and PCL ligament conditions. The images were preprocessed and annotated for object detection and classification tasks.

---

# 🎯 Problem Statement

Diagnosing knee ligament injuries from MRI scans is time-consuming and requires significant expertise from radiologists. Existing AI solutions mainly focus on classification and often lack localization, explainability, and multilingual reporting support. NeuroRad Vision addresses these limitations through AI-assisted localization, explainable visualization, and multilingual automated reporting.

---

# 🔍 Explainable AI (XAI)

The system integrates Grad-CAM visualization to highlight important anatomical regions influencing the model prediction. This improves transparency and trustworthiness of AI-assisted diagnosis in clinical workflows.

---

# 🌐 Multilingual Report Support

Using the SARVAM API, the generated clinical reports can be translated into multiple Indian languages, improving accessibility and communication across diverse healthcare environments.

---

# ⚠️ Current Limitations

- Currently supports only ACL and PCL conditions
- Limited dataset size
- Works primarily on image-form MRI inputs
- Requires larger clinical validation datasets

---

# 🔮 Future Scope

- DICOM and PACS integration
- Tear severity grading
- Meniscus and multi-ligament analysis
- Similar case retrieval system
- Prior MRI comparison
- Voice-assisted reporting
- Real-time cloud deployment

---

# 👨‍💻 Author

**Swapnil Banerjee**

---

# 📚 References

- Knee injury detection using deep learning on MRI studies: a systematic review (Diagnostics, 2022)
- Semi-automated detection of anterior cruciate ligament injury from MRI (Elsevier, 2017)
- MRI in diagnosing multiple ligament knee injuries (Springer, 2022)

---

# ⭐ Project Vision

To build an Explainable AI-assisted radiology support system that improves diagnostic efficiency, interpretability, and accessibility in orthopedic MRI analysis.