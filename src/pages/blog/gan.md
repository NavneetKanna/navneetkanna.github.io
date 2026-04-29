---
layout: ../../layouts/BlogPost.astro
title: "Generative Adversarial Networks (GANs)"
date: 2025-09-27
---

GANs are generative models: they create new data instances that resemble the training data. There are 2 models involved: Generators and Discriminators.

As the name suggests, Generators generates fake data, while Discriminator tries to distinguish between real and fake data.

Let's look at some code to understand it better. Lets use MNIST dataset.

First lets start with generator. It takes in random noise and generates as outputs a sample.

```python
z = torch.randn(batch_size, latent_size)
```

The `latent_size` is an arbitrary choosen number (usually 100). So the generator picks a sample from the latest space (usually a gaussian distribution) and learns a function that can map this sample to an actual digit.

Next, we feed the generated data to the discriminator to see what it thinks. Initially, discriminator will be able to tell correctly which data is fake or real, but as training progresses, it will find it hard and soon will start making a 50/50 choice.

```python
fake_images = G(z)
outputs = D(fake_images)
```

Now comes a very important step,

```python
g_loss = loss_func(outputs, real_labels)
g_optimizer.zero_grad()
g_loss.backward()
g_optimizer.step()
```

The loss of the generator model is calculated between the output of the discriminator and the real labels (which are nothing but 1's). During the initially stages, the discriminator correctly classifies the outputs as say all 0's, meaning it is fake data. So therefore, the loss will be very high since we are comparing it with all 1's. This is done because, we want the generator to generate data that looks as real as the dataset. In other words, the goal is to minimize this loss function, which means the generator should produce data that the discriminator thinks is real. Notice that the gradients flow through the discriminator and into the generator, but we dont update the discriminator weights (since we dont do d_optimizer.step()).

Next the discriminator. It is a binary classification model. It gets fed with both real and fake data.

```python
# Feed with real data
outputs = D(real_images)
d_loss_real = loss_func(outputs, real_labels)
real_score = outputs

# Feed with fake data
z = torch.randn(batch_size, latent_size)
fake_images = G(z)
outputs = D(fake_images.detach())
d_loss_fake = loss_func(outputs, fake_labels)
fake_score = outputs

d_loss = d_loss_real + d_loss_fake
d_optimizer.zero_grad()
d_loss.backward()
d_optimizer.step()
```

Notice that both the real and fake loss is added. This is because the discriminator should learn to classify both classes correctly and also since the data comes from 2 different sources. When training the discriminator model, we do not want to train the generator model, hence we detach the output of the generator model which ensures the gradients does not flow to the generator model.

And this is how GAN works.

--- 

References

1. https://developers.google.com/machine-learning/gan
