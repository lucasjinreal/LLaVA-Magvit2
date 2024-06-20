# 🌟 LLaVA MagVit2: Combines MLLM Understanding and Generation with MagVit2

## 📖 Introduction

Welcome to the **LLaVA MagVit2** project! This repository combines the power of Multimodal Large Language Models (MLLM) with the advanced capabilities of MagVit2 for image understanding and generation. Our main goal is to leverage the Image tokenizer inside LLava to enhance both the comprehension and creation of images, pushing the boundaries of what's possible in the field of computer vision and natural language processing.

> ⚠️👷 the repo is currently at early stage, combining Image understanding and Generation is a good exporation on Multi Modal LLMs. Our ultimate goal is to make it a mini GPT4o with vision and voice ability. 


> Our codebase based on LLaVA-Next, thanks to the original author.


## 🤖 Get Started

Now, we just able to intergrate Magvit2 into llava, you can try reconstruct the image with:

Download the magvit2 tokenizer checkpoints:

```
mkdir checkpoints && cd checkpoints
huggingface-cli download TencentARC/Open-MAGVIT2 --local-dir magvit2
```
then: 

```
# choose num_down=3 for imagenet_128_B.ckpt, choose num_down=4 for imagenet_256_B.ckpt
python test_image_tokenizer.py --ckpt_path checkpoints/magvit2/imagenet_128_B.ckpt --num_down 3 --image_file images/a.jpg
```

You will see how the text being reconstructed well in the saved image.

Next, we are going to using these features for understanding.

| Original | Reconstructed (d=3, [IN128_Base](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_128_B.ckpt)) | Reconstructed (d=4, [IN256_Base](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_256_B.ckpt)) |
| --- | --- | --- |
| ![Alt text for Image 1](images/a.jpg) | ![Alt text for Image 2](images/a_constructed_128.png) | ![Alt text for Image 2](images/a_constructed_256.png) |
| ![Alt text for Image 1](images/b.jpg) | ![Alt text for Image 2](images/b_constructed_128.png) | ![Alt text for Image 2](images/b_constructed_256.png) |
| ![Alt text for Image 1](images/c.jpg) | ![Alt text for Image 2](images/c_constructed_128.png) | ![Alt text for Image 2](images/c_constructed_256.png) |


Left is origin image, right is reconstruct with only [1, 18, h, w] codecs.


## 🏆 Results

Our approach is to yielded impressive results in various benchmarks and applications. Here are some possiable highlights:

- **Image Understanding**: Achiev state-of-the-art performance in image classification and object detection tasks.
- **Image Generation**: Generat high-quality images from textual descriptions with remarkable fidelity and diversity.
- **Multimodal Tasks**: Successfully integrat image and text modalities to perform complex tasks such as visual question answering and image captioning.


### 📊 Example Results

| Task                      | Metric   | Score |
| ------------------------- | -------- | ----- |
| Image Classification      | Accuracy | -     |
| Object Detection          | mAP      | -     |
| Image Generation          | FID      | -     |
| Visual Question Answering | Accuracy | -     |
| Image Captioning          | BLEU-4   | -     |

## 🧪 Experiment

To reproduce our experiments, follow these steps:

1. **📥 Clone the Repository**:
    ```bash
    git clone https://github.com/lucasjinreal/LLaVA-MagVit2.git
    cd LLaVA-MagVit2
    ```

2. **📦 Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **⬇️ Download Pretrained Models**:
    - Download the pretrained models for LLava and MagVit2 from the provided links and place them in the `models/` directory.

4. **🚀 Run Experiments**:
    to be done.


## 🛤️ Roadmap

We have an exciting roadmap ahead to further enhance the capabilities of LLaVA MagVit2:

- **Phase 1: Model Optimization** 🛠️
    - Fine-tune the models for specific tasks and datasets.
    - Optimize the Image tokenizer for faster and more accurate processing.

- **Phase 2: Feature Expansion** 🌐
    - Integrate additional modalities such as audio and video.
    - Develop new algorithms for more complex multimodal tasks.

- **Phase 3: Community Engagement** 🤝
    - Open-source the project and invite contributions from the community.
    - Organize workshops and challenges to foster innovation and collaboration.

- **Phase 4: Real-world Applications** 🌍
    - Deploy the models in real-world applications such as autonomous driving, healthcare, and entertainment.
    - Collaborate with industry partners to bring cutting-edge solutions to market.

## 🤝 Contributing

We welcome contributions from the community! If you're interested in contributing, please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 📧 Contact

For any questions or inquiries, please feel free to raise an issue.
