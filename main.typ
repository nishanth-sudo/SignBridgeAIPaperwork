#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [A Real-Time System for Sign Language Detection from Multilingual Speech and Text Modalities],
  abstract: [
    Communication between hearing-impaired individuals and non-signers remains a major challenge, especially in real-time scenarios like video calls. Existing sign language solutions often lack support for multilingual inputs, suffer from high latency, and are limited to specific languages such as English. Moreover, most systems either rely solely on gesture recognition or fail to incorporate voice and text formats in a unified manner.
  ],
  authors: (
    (
      name: "Senthil Kumar M",
      department: [Department of Artificial Intelligence and Data Science],
      organization: [Erode Sengunthar Engineering College],
      location: [Erode, India],
      email: "senthilesec21@gmail.com"
    ),
    (
      name: "Navaneethan AT",
      department: [Department of Artificial Intelligence and Data Science],
      organization: [Erode Sengunthar Engineering College],
      location: [Erode, India],
      email: "navaneethan1131@gmail.com"
    ),
    (
      name: "Nishanth P",
      department: [Department of Artificial Intelligence and Data Science],
      organization: [Erode Sengunthar Engineering College],
      location: [Erode, India],
      email: "nishanthp03032005@gmail.com"
    ),
    (
      name: "Sidharth S",
      department: [Department of Artificial Intelligence and Data Science],
      organization: [Erode Sengunthar Engineering College],
      location: [Erode, India],
      email: "sidharthsnair003@gmail.com"
    )
  ),
  index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction
In today's world, Communication gaps between hearing and hearing-impaired individuals are still a big challenge especially during fast-paced, real-time interactions like video calls. Most current systems either support just one language or only focus on gesture input, leaving out voice and text in other local languages. With the rise of remote communication, there’s a growing need for tools that can bridge language and accessibility gaps on the fly. But existing system supports only international language such as English, French, etc., Because of this, Indian peoples are not able to use the system. Indian Languages are not supported in the existing system 


== Paper overview

This research presents a real-time, intelligent system designed to bridge the communication gap between the hearing and the deaf or hard-of-hearing communities. The proposed solution enables the seamless translation of both spoken and written language inputs into sign language, offering a more inclusive and accessible way for individuals with hearing impairments to engage in conversations and receive information.



= Literature Review 
Communication between sign language users and non-signers has long presented a challenge, particularly in dynamic, real-world scenarios. While sign language serves as a vital tool for individuals who are speech and hearing impaired, its effectiveness is often limited by a lack of universal standards and support for multilingual interaction. Learners and users of sign language face additional difficulties due to fragmented learning tools, insufficient real-time translation systems, and the absence of comprehensive platforms that support both recognition and cross-language conversion.

Early works in sign language recognition predominantly focused on static gesture classification using traditional computer vision techniques. However, these systems struggled with dynamic gesture sequences, scalability across sign languages, and real-time processing. With the advent of deep learning, researchers began employing Convolutional Neural Networks (CNN's) and Recurrent Neural Networks (RNN's) to improve the recognition of signs, especially at the alphabet and word level. Kaur and Kaur (2020) highlighted the effectiveness of CNN's in recognizing hand gestures, while newer approaches have explored hybrid models combining CNN's with LSTMs for temporal gesture tracking. 

More recent studies have shifted towards multimodal frameworks that integrate audio, text, and gesture-based inputs. For instance, Kothadiya et al. (2022) proposed “Deepsign,” a deep learning-based system that enables real-time detection of Indian Sign Language using LSTM-GRU networks. Mistree et al. (2023) further advanced this by developing a translation pipeline from ISL to multiple Indian languages using MobileNet-V2 and Neural Machine Translation (NMT), addressing the lack of regional language support.

The use of Natural Language Processing (NLP) and Automatic Speech Recognition (ASR) in tandem with sign recognition has gained traction as well. These systems allow users to input voice or text in their native language, which is then translated and rendered as sign gestures—either via animated avatars or pre-recorded clips. This bidirectional communication model greatly benefits non-signers, enhancing inclusivity during real-time interactions such as video calls.

Despite these advancements, current systems often lack features critical for learners, such as accurate feedback, multilingual guidance, and voice output. There remains a noticeable gap in platforms that facilitate the learning and usage of sign language in a practical, real-world setting. Most tools focus on either learning support or translation, but rarely both in a seamless interface.

This study aims to fill that gap by integrating alphabet and word recognition, text-to-action conversion, multi-language support, and voice output into a unified application. Through the adoption of advanced machine learning algorithms and rigorous software design, the proposed system strives to not only assist in communication but also elevate the learning experience for new and existing users of sign language.


= Methodology <sec:methods>
== System Components

A key feature of this system is its robust support for multiple languages. It is capable of processing speech and text inputs in a wide range of languages including regional Indian languages like Tamil, Hindi, Malayalam, Telugu, Kannada, and Tulu, as well as English and other widely spoken international languages. This ensures that the system can cater to users from diverse linguistic backgrounds, making it highly versatile and globally applicable.

=== System Components

The system architecture is built on several advanced technologies that work in harmony to deliver real-time sign language translation:

==== Speech-to-Text Module
At the front end of the pipeline, a powerful speech recognition engine—OpenAI’s Whisper—is utilized. This state-of-the-art model excels at transcribing audio inputs with remarkable accuracy, even in noisy environments or across various dialects. Whether a user is speaking in Tamil or English, the Whisper model ensures that the spoken words are accurately captured and converted into text.

==== Text Processing and Natural Language Understanding
Once the speech is transcribed (or a text input is directly provided), the content undergoes several NLP-based transformations. This includes text normalization (removing unnecessary elements), grammar correction, semantic interpretation, and context extraction. These processes ensure that the intended meaning of the message is preserved and appropriately prepared for sign language conversion.

==== Language Identification and Routing
A dedicated language detection layer is integrated into the pipeline to identify the language of the input, especially when it comes from text sources where no audio is available. This ensures that the appropriate linguistic and cultural context is applied before translation, which is crucial for accurate sign representation.

==== Sign Language Generation
The final processed output is translated into sign language gestures. This can be achieved in two ways:

===== Animated Avatars 
3D avatars or digital characters perform the corresponding signs, mimicking human-like movements.

===== Gesture Video Snippets 
Alternatively, pre-recorded video clips of human signers demonstrating the gestures can be played in sequence.

This dual approach ensures both flexibility and realism, allowing developers to choose the most suitable output method based on user needs and available resources.


== Data Collection
 Data is collected through webcam and stored manually in a database. These image data is not only collected from webcam alone but inorder to train with high accuracy, is used from a self generated dataset. This dataset is used to process/train so that it can also act as a multilingual dataset format. This dataset ensures that the input can be given in almost any language. 


 #align(center,image("recording-off.png",width: 75%))

==== i) 
Data not being collected when the Recording 
is turned 'OFF'
 

#align(center,image("recording-on.png", width: 75%))

==== ii)
Data is being collected when the Recording is turned 'ON'





 

== Computer Vision Algorithms used:

=== MediaPipe Hand Tracking

MediaPipe helps computers see and understand your hands in real-time using just your regular camera. It spots your palm, then maps 21 key points on your fingers and wrist. This lets your computer know what gesture you're making, instantly and accurately.


Begin

  Step 1: Capture real-time video frame from webcam.
  
  Step 2: Convert the frame to RGB format (required by MediaPipe).
  
  Step 3: Pass the frame to MediaPipe Hands module.
  
  Step 4: Detect 21 hand landmarks per detected hand.
  
  Step 5: Extract landmark coordinates (x, y, z).
  
  Step 6: Normalize and feed coordinates to gesture recognition model (e.g., CNN/LSTM).

End

=== Convolutional Neural Networks (CNN)

CNN's are like eyes for the computer. They scan images bit by bit, learning to recognize shapes, edges, or patterns—like how we recognize letters or faces. They're great at helping your system understand what hand sign or alphabet a person is showing.


Begin

  Step 1: Input preprocessed image of hand sign (e.g., grayscale, resized).
  
  Step 2: Pass through multiple convolutional layers to extract features.
  
  Step 3: Apply activation function (ReLU) after each convolution.
  
  Step 4: Use max pooling to reduce feature map size.
  
  Step 5: Flatten feature maps into a vector.
  
  Step 6: Pass through fully connected layers for classification.
  
  Step 7: Output class label (alphabet/word) with highest confidence.

End


=== CNN + LSTM

This hybrid model combines CNNs for spatial feature extraction from video frames with Long Short-Term Memory (LSTM) networks for capturing temporal dynamics across multiple frames—ideal for recognizing dynamic gestures or sign sequences.


Begin

  Step 1: Capture video frames of gesture sequence.

  Step 2: For each frame, Detect hand region and Extract spatial features using CNN.
  
  Step 3: Aggregate feature vectors over time into a sequence.
  
  Step 4: Feed the sequence into an LSTM network to capture temporal patterns.
  
  Step 5: Output final gesture class based on LSTM output.

  End

== Speech Recognition

=== Wav2Vec 2.0


Begin

  Step 1: Input raw audio waveform (16kHz).

  Step 2: Pass waveform through feature encoder (convolutional layers).
  
  Step 3: Extract latent audio representations.
  
  Step 4: Apply context network (Transformer) for sequential modeling.
  
  Step 5: Predict phoneme/token probabilities using CTC (Connectionist Temporal Classification).
  
  Step 6: Decode final text output using language model.

End


=== MarianMT / IndicTrans2

Wav2Vec 2.0 is a self-supervised deep learning model developed by Facebook AI that learns speech representations directly from raw audio waveforms, allowing accurate speech-to-text conversion even with limited labeled data.


Begin

  Step 1: Input source sentence in language A.

  Step 2: Tokenize input using sentencepiece or BPE tokenizer.
  
  Step 3: Encode tokens using encoder (Transformer-based).
  
  Step 4: Generate context embeddings.
  
  Step 5: Decode embeddings using decoder to generate tokens in language B.
  
  Step 6: Detokenize output to form translated sentence.

End

== Sign Language Generation / Mapping

=== Rule-Based Sign Mapping

A deterministic approach where specific signs are associated with predefined words or sentences, often using a database or lookup table. This is used to map translated text into sign language videos or avatar movements.

Begin
  
  Step 1: Receive recognized text from input (speech or typed).
  
  Step 2: Tokenize text into individual words.
  
  Step 3: For each word, Search gesture dictionary/video snippet/avatar mapping.
  
  Step 4: Concatenate gesture videos or play avatar animations in sequence.
  
  Step 5: Render sign output to user.

End


=== Deep Q-Learning

DQN is a reinforcement learning algorithm that learns optimal actions by approximating the Q-value function using deep neural networks. It's useful for systems that adapt or personalize over time based on feedback.



Begin
 
  Step 1: Initialize Q-network with random weights.
  
  Step 2: For each interaction:
            - Observe current state (user profile, past gestures, context).
            - Choose action (gesture/video/response) using epsilon-greedy strategy.
            - Observe reward and next state.
            - Store experience in replay buffer.

            Step 3: Sample mini-batch from buffer.
  
            Step 4: Compute Q-value targets and train network using loss minimization.
  
            Step 5: Update network weights and repeat.
End

== Text-to-Speech (TTS)

=== Tacotron 2

Tacotron 2 is a deep neural network architecture for end-to-end text-to-speech synthesis. It converts input text to a spectrogram and uses WaveNet (or another vocoder) to produce natural-sounding speech.


Begin

  Step 1: Input raw sentence text.

  Step 2: Convert text to phoneme sequence.
  
  Step 3: Encode phoneme sequence using encoder RNN.
  
  Step 4: Generate mel-spectrogram using decoder RNN + attention.
  
  Step 5: Use WaveGlow / Vocoder to convert mel-spectrogram to audio waveform.
  
  Step 6: Output natural-sounding voice in selected language.

End



== Model Training
 Training is done using GAN (Generative Adversial Network) Model. This encourages the input images to be converted into trained model that the computer 

= Implementation Details

This project implements a Generative Adversarial Network (GAN) using the PyTorch deep learning framework to generate realistic sign language gesture images. The GAN architecture is trained on a dataset of Indian Sign Language images, aiming to synthesize plausible gesture visuals conditioned on simple text prompts.

== 1. Library Dependencies and Setup

The implementation begins by importing necessary libraries:

- `torch`, `torch.nn`, `torch.optim`: For defining and training neural network models.
- `torchvision`: For image transformations, dataset utilities, and saving output images.
- `os`, `gc`, `random`: For system management, garbage collection, and randomness.
- Custom modules: `models.generator`, `models.discriminator`, and `utils.preprocessing` for data loading and embedding conversion.

These tools collectively enable efficient GPU-based training using PyTorch’s dynamic computation graphs.

== 2. Generator Architecture

The Generator ($G$) is a neural network designed to map a latent noise vector (or an embedded text representation) into a 64×64 image mimicking a real sign language gesture. It follows a typical Deep Convolutional GAN (DCGAN) style:

- Input: Random noise vector or text embedding $z$
- Layers: A series of `ConvTranspose2d` (transposed convolutions) to upsample, followed by `BatchNorm2d` and `ReLU` activations.
- Output: A 3-channel image passed through `Tanh` activation to normalize pixel values between [-1, 1].

== 3. Discriminator Architecture

The Discriminator ($D$) is a convolutional neural network trained to classify input images as either real (from the dataset) or fake (from the Generator):

- Input: 3-channel image of size 64×64
- Layers: Multiple `Conv2d` layers with `LeakyReLU` activations and `BatchNorm2d`
- Output: A scalar probability through a `Sigmoid` function indicating the "realness" of the image

== 4. Loss Function and Optimizers

The models are trained using Binary Cross Entropy (BCE) Loss:

- Discriminator Loss:

  $$
  \mathcal{L}_D = - \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]
  $$

- Generator Loss:

  $$
  \mathcal{L}_G = - \mathbb{E}_{z \sim p_z}[\log D(G(z))]
  $$

These losses optimize the adversarial objective using the Adam optimizer, with a learning rate of 0.0002 and momentum parameters $(\beta_1 = 0.5, \beta_2 = 0.999)$.

== 5. Data Loading and Preprocessing

The dataset comprises real sign language gesture images. Images are:

- Resized to 64×64 pixels
- Converted to tensors
- Normalized to have pixel values in [-1, 1]

A `DataLoader` batches the dataset for efficient GPU training.

 python
transform = transforms.Compose([
  transforms.Resize(64),
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])



#align(center,image("train1.png",width: 75%))

  i) Model Training using GAN

#align(center,image("train2.png",width: 75%))

  ii) Trained Model Sample Output

  
#align(center,image("train3.png", width: 75%))

  iii) Alphabets getting trained in Sign Language for Prediction


= Testing And Outcome

 During the testing phase, the system uses a webcam to capture real-time video, allowing users to interact naturally with hand gestures. These gestures are then recognized by a machine learning model that has been trained beforehand.

The implementation starts by importing a few essential libraries:

OpenCV (cv2) is used to access the webcam and display the video feed.

NumPy helps with efficient numerical operations on arrays.

MediaPipe provides powerful hand tracking and landmark detection tools.

Joblib is used to load the pre-trained classifier and label encoder.

Pandas helps in organizing landmark data into a format suitable for model input.

The trained model, stored as a .pkl file, is loaded into the program. This model can be any common classifier like a Random Forest or Support Vector Machine (SVM) that has been trained to identify gestures based on the positions of hand landmarks. The label encoder is also loaded to convert the model’s numeric predictions into meaningful gesture names.

Using MediaPipe’s Hands module, the system can detect up to two hands in a video frame and extract 21 landmark points per hand, each with three coordinates (x, y, z). If only one hand is detected, the system fills in zeros for the second hand, ensuring the input size remains consistent (126 values total).

For every frame from the webcam:

It is cropped into a square shape,

Resized to 480×480 pixels,

Flipped horizontally (for a mirror-like interaction), and

Converted to RGB format before being processed.

If a hand is detected, the landmarks are flattened into a list, converted to a DataFrame, and passed to the model for prediction. The predicted output is then decoded using the label encoder and displayed as text directly on the video feed.

If no hands are detected, the message “No hand detected” is shown on the screen. This process runs in real-time and continues until the user presses the 'q' key to exit.

The system is also designed to handle errors—such as issues with loading the model or making predictions—ensuring smooth and uninterrupted performance even in varied lighting or background conditions.

#box(image("op.png", width: 40%)) #box(image("op (2).png", width: 40%))


==== predicted images (output) given above




= Conclusion & Future Scope

In this project, we developed a real-time system that recognizes and interprets Indian Sign Language (ISL) gestures using a standard webcam and machine learning techniques. By combining MediaPipe's reliable hand tracking capabilities with a trained gesture classification model, the system can detect hand movements and convert them into readable text almost instantly. This interactive setup enhances communication for individuals who are speech or hearing impaired, allowing for more inclusive and accessible interactions. Live testing showed that the system performs well across different users and environments, and its modular design means it can be extended easily to support additional input methods like speech or text.

While the system currently focuses on static gesture recognition, there are many exciting directions to explore in the future:

- Recognizing Dynamic Gestures: To support full sentences and natural sign language, we aim to include models like LSTMs or 3D CNNs that can understand sequences of movements over time.
- 
- Multilingual Gesture Translation: Expanding the system to output text in regional Indian languages can help reach a broader group of users and break more communication barriers.
- 
- Gesture Output from Voice or Text: A future version could take spoken or typed input and translate it into sign language using avatars or generated gesture videos, making it a two-way tool.
- 
- Mobile Accessibility: By optimizing the system for mobile and edge devices, users could interact with it directly from their smartphones without needing an internet connection.
- 
- Emotion and Context Awareness: Adding the ability to detect emotions and context from user inputs could make the system’s responses more natural and meaningful.
- 
- 3D Avatar Integration: Instead of showing only text, a 3D avatar could perform the signs visually, making it easier for people who don't read text well or are more visually oriented.

These future improvements aim to create a more complete and user-friendly tool that can truly assist in everyday communication for the deaf and hard-of-hearing community.
