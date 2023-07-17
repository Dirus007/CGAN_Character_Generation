# CGAN_Character_Generation
The project uses CGAN (Conditional Generative Adversarial Network) model on dataset EMNIST which contains handwritten characters (digits and alphabets). The model aims to replicate these realistic handwritten characters.
The dataset used is the balanced dataset from https://www.kaggle.com/datasets/crawford/emnist

The starting of code is from main.py

I was unable to upload extracted .csv file due to size limit of Github

# Working
I imagine a GAN's model as a competition between an artwork Appraiser (the person whose job is check if a painting is real and appraise its value) and a Forger (who makes fake paintings and wants to make it less distinguishable from real paintings).

Appraiser is the Discriminator model and Forger is the Generator model in GAN. The Appraiser is learning and knows some basics of distinguishing a fake from real but he also learns some new things while working.

The Forger tries to make real paintings and copies different artists. When he fails, he changes his artstyle a little. That is, he improves himself.

In this continous cycle, both the Forger and Appraiser become better at job. Forger makes indistinguishable paintaings of any artists he likes, and Appraiser finds the fake.

After some time Forger has become so good at his work, that he himself has become a very good artist. He can now replicate other people's work. Our Forger became an expert artist.

This is how Generator model generates realistic fakes.

https://www.cloud-science.de/wp-content/uploads/2022/05/GENERATIVE-ADVERSARIAL-NETWORK-GAN-Cartoon-web.png
