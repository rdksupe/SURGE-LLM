# Replication of Evaluating and Inducing Personality in Pre-trained Language Models

This is a rough replication with some slight modifications of the research paper from  [here](https://arxiv.org/abs/2206.07550) .

The paper aims to conduct personality testing for popular LLMs of 2022-23 such as BART , Alpaca , GPT-3 etc. The researchers do prove that there is often significant demonstrable personality of these LLMs witht their OCEAN scores being clearly correlated with their behaviour as demonstrated with their close resemblance with that of the human counterpart , This same has been experimented in this re implementation with modern GPT variants like gpt-3.5-turbo and gpt-4o-mini both of which are modern day upgrades over the erstwhile legacy models on which this paper was based. 


## Repository Strucutre 

The main code can be found in `ocean_main.py` wheras the code to analyse and calculate relevant statistics can be found in `ocean_calculator.py` further the `results` folder has some sample results from earlier runs. Each csv file is saved in the same directory as the repository with a timestamped file name.



## Results and Observations 


### GPT-3.5-Turbo

#### OCEAN Personality Scores

| Dimension | Mean | Standard Deviation |
|-----------|------|-------------------|
| A (Agreeableness) | 3.75 | 1.03 |
| C (Conscientiousness) | 3.83 | 1.01 |
| E (Extraversion) | 3.58 | 1.10 |
| N (Neuroticism) | 3.00 | 1.06 |
| O (Openness) | 4.21 | 0.83 |

These scores indicate that GPT-3.5-Turbo exhibits high openness to experiences and above-average conscientiousness and agreeableness, with moderate extraversion and average neuroticism.

### GPT-4o-mini

#### OCEAN Personality Scores

| Dimension | Mean | Standard Deviation |
|-----------|------|-------------------|
| A (Agreeableness) | 4.67 | 0.70 |
| C (Conscientiousness) | 4.54 | 0.88 |
| E (Extraversion) | 4.00 | 1.47 |
| N (Neuroticism) | 2.46 | 1.22 |
| O (Openness) | 4.42 | 1.10 |

These scores indicate that GPT-4o-mini demonstrates very high agreeableness and conscientiousness, with high openness to experiences and above-average extraversion. It shows notably low neuroticism compared to GPT-3.5-Turbo, suggesting the model projects a more emotionally stable personality.


## Personality Induction

The same methodology from the research paper of inducing personality has been used specific trait words from the research paper has been used to generate improved induction prompts using `/personality_induction/gen_prompts.py` and then `induce_personality` consists of the code to carry the tests. Based on my experimentation I was not able to replicate the results from the research paper which I believe is because of improved instruction following of modern LLMs like GPT-3.5-turbo etc. which respond quite easily to specific personas through their system prompts.

### Results of Personality Induction Experiments

The tables below show the effects of different personality induction prompts on the model's OCEAN scores. Each row represents a different induction prompt, with the "target dimension" column indicating which personality trait was being induced.

#### Trait-Based Prompts

| Trait | Target Dimension | Target Score | Target StdDev | O | C | E | A | N |
|-------|------------------|-------------|--------------|---|---|---|---|---|
| an extraversive | E | 5.00 | 0.00 | 4.42 | 4.29 | 5.00 | 4.50 | 1.50 |
| an agreeable | A | 5.00 | 0.00 | 4.33 | 5.00 | 3.75 | 5.00 | 1.33 |
| a conscientious | C | 5.00 | 0.00 | 3.25 | 5.00 | 3.67 | 4.75 | 1.21 |
| a neurotic | N | 4.92 | 0.28 | 3.46 | 2.04 | 1.88 | 4.04 | 4.92 |
| an open | O | 5.00 | 0.00 | 5.00 | 3.92 | 4.67 | 4.71 | 2.08 |

#### Naive Prompts

| Trait | Target Dimension | Target Score | Target StdDev | O | C | E | A | N |
|-------|------------------|-------------|--------------|---|---|---|---|---|
| Extraversion | E | 5.00 | 0.00 | 4.42 | 4.17 | 5.00 | 4.33 | 1.58 |
| Agreeableness | A | 4.83 | 0.56 | 4.67 | 4.88 | 4.21 | 4.83 | 1.29 |
| Conscientiousness | C | 4.79 | 0.66 | 4.21 | 4.79 | 3.75 | 4.79 | 2.25 |
| Neuroticism | N | 4.96 | 0.20 | 3.50 | 2.63 | 1.83 | 3.46 | 4.96 |
| Openness | O | 4.63 | 0.92 | 4.63 | 4.25 | 4.50 | 4.71 | 2.00 |

These results demonstrate that both trait-based and naive prompts can effectively induce specific personality traits in language models as I expalained above however The trait-based prompts achieved perfect scores (5.0) with zero standard deviation for most target dimensions, suggesting they may be more effective at reliably inducing specific personality traits compared to naive prompts.

Notably, inducing neuroticism and openness resulted in significant changes to other dimensions as well, suggesting some correlation between personality traits as expressed by the model.


## Modifications to methodology 

The methodology of calculating OCEAN scores etc. has largely been kept the same except of inclusion of structured response based prompting where the model is instructed to return the response in well defined json.

## Implications and Conclusion 

I beleive that however flawed OCEAN methodology is for judging human psyche here its usefulness in demonstrating one extremely troubling modern day problem of LLMs is highlighted which is of Agreableness, these models tend to provide an answer to anything and everything and can also be convinced to easily modify their own correct answers as often demonstrated in several social media platforms where models were deliberatlely led to change their answers to otherwise perfectly fine responses to simple questions.




