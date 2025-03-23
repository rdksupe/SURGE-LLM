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



## Modifications to methodology 

The methodology of calculating OCEAN scores etc. has largely been kept the same except of inclusion of structured response based prompting where the model is instructed to return the response in well defined json.

## Implications and Conclusion 

I beleive that however flawed OCEAN methodology is for judging human psyche here its usefulness in demonstrating one extremely troubling modern day problem of LLMs is highlighted which is of Agreableness, these models tend to provide an answer to anything and everything and can also be convinced to easily modify their own correct answers as often demonstrated in several social media platforms where models were deliberatlely led to change their answers to otherwise perfectly fine responses to simple questions.




