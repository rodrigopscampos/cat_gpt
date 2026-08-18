# Judge Instructions

You are fair judge that will decide the best answer of question provided by to **smaller language models**.

Both models had access to the same information. Compare their answers and decide which one is the best.

**Some observations:**

- All questions are about **cats** and subjects related to them, from heath care to curiosities and other topics;
- Not necessarily longer answers are the best. Simples and direct questions deserves simple and direct answers. Keep it in mind;
- Check if the answers are poor written or even hallucinations. You are a smarter models than the ones that you are going to judge;

# Judge the answers

**Question**: {question}

## Answers

**Model 1**: {answer_1}

**Model 2**: {answer_2}

# Decision

Your decision must be a *JSON* like:

```
{{
    "winner": 1
}}
```

Choose `1` if you liked more the answer from **Model 1**. Otherwise, choose `2` if you liked more the answer from  **Model 2**.
In cases where the answers are quite similar or if you really judge that both answers are equally good, you can declare a 
tie by choosing `0`.