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

What do you hear when you walk into a fast food joint during the peak hours of lunch time? Noise, noise, and more noise. It is hard to even think with everyone taking orders, the chefs screaming at eachother in the back, and all the chattering happening. Its as if everything is amplified x5!

This is the problem with modern hearing aids. It aims to amplify everything instead of only what is important. We want to hear meaningful things like friends and loved one not everything all at once. This is what this project aims to do. We apply deep learning to noise so that it removes the unnecessary noise and keeps the important ones. We do this through a deep learning technique called auto encoders.

<a>
    <img src="https://miro.medium.com/max/600/1*nqzWupxC60iAH2dYrFT78Q.png" alt="Logo" width="1100" height="500">
 </a>


As you can see from above, what the auto encoder does is compress the image and then reconstructs that same image. You might be asking what is the point of that? Noise can be viewed as a image called a spectrogram. If you are in a resteraunt the spectrogram will have a lot of noise in the image. The auto encoder aims to take in that noisy spectrogram and reconstruct it so that the noise has been removed from the spectrogram.

This is the goal of this project.
* Take in a noisy spectrogram from a loud environment
* Apply deep learning so that it removes the noise
* Return a clean spectrogram that only has noise of what you care about

### Built With


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







<!-- ACKNOWLEDGEMENTS -->
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
