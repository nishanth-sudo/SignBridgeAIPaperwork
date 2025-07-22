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
@netwok2020 @netwok2022.

== Paper overview

This paper proposes a real-time system for sign language detection from multilingual speech and text inputs, aimed at enhancing communication accessibility for the deaf and hard-of-hearing community. The system is designed to interpret both spoken audio and typed textual inputs in multiple languages—including Tamil, Hindi, Malayalam, Telugu, Kannada, Tulu, English, and several international languages—and convert them into sign language gestures using either animated avatars or gesture video snippets.

The core components of the system include a speech-to-text module powered by OpenAI Whisper, which efficiently transcribes spoken content from diverse languages with high accuracy. This output is further processed using natural language processing (NLP) techniques to normalize, clean, and contextualize the content. A language detection layer ensures that the correct linguistic model is applied before converting the processed input into a series of mapped signs.


= Methods <sec:methods>
#lorem(45)

$ a + b = gamma $ <eq:gamma>

#lorem(80)

#figure(
  placement: none,
  circle(radius: 15pt),
  caption: [A circle representing the Sun.]
) <fig:sun>

In @fig:sun you can see a common representation of the Sun, which is a star that is located at the center of the solar system.

#lorem(120)

#figure(
  caption: [The Planets of the Solar System and Their Average Distance from the Sun],
  placement: top,
  table(
    // Table styling is not mandated by the IEEE. Feel free to adjust these
    // settings and potentially move them into a set rule.
    columns: (6em, auto),
    align: (left, right),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0  { rgb("#efefef") },

    table.header[Planet][Distance (million km)],
    [Mercury], [57.9],
    [Venus], [108.2],
    [Earth], [149.6],
    [Mars], [227.9],
    [Jupiter], [778.6],
    [Saturn], [1,433.5],
    [Uranus], [2,872.5],
    [Neptune], [4,495.1],
  )
) <tab:planets>

In @tab:planets, you see the planets of the solar system and their average distance from the Sun.
The distances were calculated with @eq:gamma that we presented in @sec:methods.

#lorem(240)

#lorem(240)
