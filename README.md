<div id="top"></div>

[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="https://raw.githubusercontent.com/mrbraden56/Best-README-Template/master/images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <p align="center">
    Audio Denoiser 
    <br />
  </p>
</p>




<!-- ABOUT THE PROJECT! -->
## About The Project

<a>
    <img src="https://miro.medium.com/max/1182/1*OOTqBsjpuXyfYJVdPxWtBA.png" alt="Logo" width="1100" height="500">
  </a>

This is an implementation of <a href="https://arxiv.org/pdf/1609.07132.pdf">A Fully Convolutional Neural Network for Speech Enhancement</a>. This research paper presents a method for speech enhancement that can enhance the quality of understanding speech for hearing aid users. As input, the model takes 8 frames of a noisy spectrogram and outputs the corresponding clean frame. This way the clean frame will have the time and frequency dependencies of the current frame and the past 7.

The biggest challenges faced in this project was understanding how to efficiently train and evaluate a model. As a college student, I do not have access to 100 GPU's so I had to be very methodical on how I approached this. The method I found most useful was to purposely overfit the model on 1 training example, that way I knew the model worked as it should. Then from there I incrementally increased the amount of data until I found the result that brought me the best model loss.

The model I trained in this repo was also used in a web app for speech enhancement in my senior design class and you can see that <a href="https://github.com/jrhaxton/JayHear">here</a>.

<!-- RESULTS! -->
## Results
<p align="center">
  <a>
      <img src="Audio/experiment25.png" alt="Logo" width="700" height="500">
    </a>
  </p>
  Here you can see a spectrogram of the clean audio, clean audio with noise added onto it, and finally the denoised audio. You can see in the denoised audio that the audio with the highest db have great reconstruction, however, the lower the db is the worse the reconstruction gets. You can compare the noisy and denoised audio files <a href="https://mrbraden56.github.io/#/audiofiles">here</a>.

 <audio controls="controls">
  <source type="audio/wav" src="Audio/noisy12006_SNRdb_0.0_clnsp12006.wav"></source>
</audio>

<!-- WHAT I'D DO DIFFERENTLY! -->
## What I'd do Differently
One of the most imporant lessons I learned in this project was to make a plan. From the start to finish of this project, I probably deleted everything and restarted 3 times. This was due to various reasons including bugs, incorrect feature extraction, and misunderstanding of the machine learning model. If at the begenning I would have laid out a plan such as creating a program flowchart and understanding different loss functions, it would have cut my time in half.  

<!-- BUILT WITH! -->
## Built With


<a href="https://librosa.org/doc/main/index.html">
    <img src="https://librosa.org/doc/main/_static/librosa_logo_text.svg" alt="Logo" width="100" height="100">
       <br />
 </a>
 
 
 <a href="https://numpy.org/doc/stable/index.html#">
    <img src="https://numpy.org/doc/stable/_static/numpylogo.svg" alt="Logo" width="100" height="100">
       <br />
 </a>
      
  <a href="https://pytorch.org/">
    <img src="https://pytorch.org/assets/images/pytorch-logo.png" alt="Logo" width="100" height="100">
       <br />
 </a>








## Acknowledgements
* [Sthalles](https://sthalles.github.io/practical-deep-learning-audio-denoising/#:~:text=Introduction,degrading%20the%20signal%20of%20interest.&text=In%20this%20situation%2C%20a%20speech,to%20improve%20the%20speech%20signal.)
* [othneildrew github](https://github.com/othneildrew)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/braden-lockwood-b7606a1b5/
[product-screenshot]: images/screenshot.png
