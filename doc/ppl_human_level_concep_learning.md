# Title

[**Human-level concept learning through probabilistic program induction**](http://www.sciencemag.org/content/350/6266/1332.abstract). Science, 350(6266), 1332-1338.
Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015).<br />
This paper belong to the topic [Probabilistic programming language](ppl.md) in [paper-on-AI](../README.md) repository.

## Motivation

 **People**<br />
  -Generalize successfully from just a single example to learn new concepts<br />
  -Use learned concepts in richer ways—for action, imagination, and explanation

 **Machine learning algorithms**<br />
  -Require tens or hundreds of examples to perform with similar accuracy

Given a single object, human can:<br />
  -Classify new examples,<br />
  -Generate new examples of similar type,<br />
  -Parse it into parts and understand their relation.

![Examples](../fig/ppl_human_level_concep_learning_fig1.png)


## Bayesian Program Learning

The paper introduces the Bayesian Program Learning (BPL) framework, which allows an algorithm to obtain the abilities described above. The BPL framework is based on three fundamental ideas:<br />
 -Compositionality<br />
 -Causality<br />
 -Learning to learn<br />
![ ](../fig/ppl_human_level_concep_learning_fig4.png)

### Example with Hand Written Characters
Characters can be parsed based on strokes initiated by pressing a pen down and terminated by lifting it up (defined as “part” in figure below). Then, each stroke can be further separated by brief pauses of pen (“subpart” in figure below)<br />
If a character, “B”, is given:<br />
 -“B” can be parsed into two parts: one stick and another with two curves(**compositionally**).<br />
 -The second part can be further broken into a set of two half circles(**Causality**).<br />
<br />
![ ](../fig/ppl_human_level_concep_learning_fig2.png)

 -Analyzing “B” learns new, primitives and relations, which can be later used to learn other characters easily(**Learning to learn**).

![ ](../fig/ppl_human_level_concep_learning_fig3.png)

## Results

### Comparision with deep learning model on one-shot learning and generative tasks

![ ](../fig/ppl_human_level_concep_learning_fig6.png)

### Generating new examples

![ ](../fig/ppl_human_level_concep_learning_fig5.png)

### Generating new concepts

![ ](../fig/ppl_human_level_concep_learning_fig7.png)


## Codes
- BPL model for one-shot learning: https://github.com/brendenlake/BPL

## Blog

- One Shot Generalization: https://casmls.github.io/general/2017/02/08/oneshot.html


