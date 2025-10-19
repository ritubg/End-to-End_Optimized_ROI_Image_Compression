# **MATLAB based End-to-End Optimized ROI Image Compression**

## **Description**

This project implements an end-to-end optimized Region of Interest (ROI) image compression system designed for face recognition tasks. Unlike traditional methods that treat ROI prediction and coding as separate steps, this approach optimizes both simultaneously for truly efficient output.

**Implementation Language: MATLAB**

---

## **Dataset**

**LFW - People (Face Recognition)**
- **Source:** Kaggle.com
- **Steps:**
  1. Download the dataset from Kaggle
  2. Extract the zip file
  3. Rename the extracted folder to **`faces`**
  4. Place all project files in the same directory as the `faces` folder

---

## **STEPS :**

1. Download all the files in the same folder where your dataset is present - **faces**
2. Run the **demo.m** file
   - Output Trained model: trainedFaceModel.mat 

3. Run the **testImageROI.m** file
   - Upload a test image in the same folder called **""face.png""**
   - Output : Original Image, ROI mask, Reconstructed Image


## **Problem Statement**

### **Traditional Image Compression Limitations**
- **Equal Treatment:** All image regions are compressed equally
- **Bit Wastage:** Important regions (like facial features) receive the same bit allocation as background
- **Sequential ROI Processing:** 
  - ROI prediction performed separately
  - ROI coding performed as independent step
  - Both optimized independently, hence output is not truly optimized
  - **Result:** Suboptimal overall compression

---

## **Proposed Solution**

ROI prediction and coding are done simultaneously, so the output is truly optimised.

---

## **Technical Architecture**

### **Encoder subnetwork :**

#### **STEP 1: Multiscale Forward Transform & Implicit ROI Prediction**

**Layers in the subnetwork:**
- Input resolution downsampled to (H/16 × W/16)
- **GDN** (Generalized Divisive Normalization) layer - to reduce noise and redundancy in the data
- Output : **F**

**Multi scale transformation:**
- **Fb:** Basic quality features (128 channels) - allocated to background regions
- **Fh:** High quality features (320 channels) - allocated to ROI regions

**Implicit ROI Prediction:**
- ROI mask predicted automatically during learning
- Output: 3D binary mask **Q**
- Mask used for rate allocation (next step)

#### **STEP 2: ROI-Based Rate Allocation**

- Element filtering operation applied
- **Rule:** If Q = 1, encode the corresponding elements
- Ensures higher bit allocation to ROI regions
- Lower bit allocation to background regions

---

### **Decoder subnetwork**

**Reconstruction :**
- **IGDN** (Inverse Generalized Divisive Normalization)
- Inverse multiscale transform network
- Reconstructs the original image with ROI optimized quality

---


## **Advantages**

1. **End-to-end optimization:** Joint training of ROI prediction and compression
2. **Automatic ROI detection:** No manual annotation required
3. **Efficient bit allocation:** More bits to important regions (faces), fewer to background
4. **Quality preservation:** Maintains high quality in regions of interest
5. **Optimized performance:** Better rate-distortion trade-off compared to traditional methods

---


## **Please read :**

- Ensure sufficient disk space for the dataset and model files
- The model is specifically trained on face images and may not perform optimally on other image types
- Increase the number of epochs in **demo.m** for better training

---

## **Citation**

This is an independent **research paper implementation**.
If you use this code or methodology in your academic work,  please also cite the original research paper:
```bibtex
@article{cai2020end,
  title={End-to-End Optimized ROI Image Compression},
  author={Chunlei Cai, Li Chen, Xiaoyun Zhang, and Zhiyong Gao},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={3442--3457},
  year={2020},
  publisher={IEEE}
}
```
---

## **Acknowledgments**

- Original paper authors: Chunlei Cai, Li Chen, Xiaoyun Zhang, and Zhiyong Gao
- LFW (Labeled Faces in the Wild) dataset creators
- IEEE Transactions on Image Processing, VOL 29, 2020

---


