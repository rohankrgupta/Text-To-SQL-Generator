# Text-To-SQL-Generator
## Project Overview

This project focuses on developing a robust system capable of translating natural language questions into executable SQL queries. The core of this project is a `facebook/bart-base` model, fine-tuned on a domain-specific dataset to understand the relationship between user questions and database schemas.

The primary goal was not just to build a model, but to establish a rigorous evaluation framework to quantitatively measure its performance, setting a strong baseline for future iterations and comparisons.

**Live Model on Hugging Face Hub:** [rkgupta3/bart-base-text-to-sql-full](https://huggingface.co/rkgupta3/bart-base-text-to-sql-full)

---

## Tech Stack

* **Modeling:** PyTorch
* **Core Libraries:** Hugging Face `transformers`, `datasets`, `accelerate`
* **Database for Evaluation:** Python's built-in `sqlite3`
* **Environment:** Google Colab (with T4 GPU)

---

## Dataset

The model was trained and evaluated using the [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) dataset. This dataset is particularly well-suited for the task as each sample is self-contained, providing:
* A natural language question (`text`).
* The corresponding ground-truth SQL query (`sql`).
* The complete database context, including `CREATE TABLE` and `INSERT` statements (`sql_context`).

A shuffled split was created for a robust and unbiased evaluation:
* **Training Set:** 5,000 examples
* **Test Set:** 1,000 examples (held out from the model during training)

---

## Methodology

The project followed a structured, end-to-end machine learning workflow.

### 1. Data Preprocessing

The key to success for this task is providing the model with the right context. The input was formatted as a single string to teach the model how to map the schema and question to a query:


Schema: [DATABASE SCHEMA] | Question: [NATURAL LANGUAGE QUESTION]


This entire string was fed into the BART tokenizer as the input, while the ground-truth SQL query served as the target label.

### 2. Model Fine-Tuning

The `facebook/bart-base` model was fine-tuned for 3 epochs using the Hugging Face `Trainer` API. The training process was configured to:
* Use a batch size of 8.
* Evaluate performance on the test set at the end of each epoch.
* Automatically save the best-performing checkpoint based on the validation loss.



### 3. Evaluation

To get a true measure of the model's ability, a simple string comparison of SQL queries is insufficient. The gold-standard metric, **Execution Accuracy**, was implemented.

The evaluation script iterates through all 1,000 examples in the held-out test set and performs the following for each:
1.  **Generate Predicted SQL:** The model generates a SQL query based on the question and schema.
2.  **Create Live Database:** A temporary, in-memory SQLite database is created.
3.  **Populate Database:** The `sql_context` (containing `CREATE` and `INSERT` statements) is executed, creating a fully populated database for the query.
4.  **Execute Both Queries:** Both the `predicted_sql` and the `ground_truth_sql` are executed against this live database.
5.  **Compare Results:** The fetched results from both queries are converted to unordered sets to handle differences in row ordering. If the sets are identical, the prediction is marked as **correct**. Any SQL error during prediction execution is marked as **incorrect**.

---

## Results & Key Findings

After a full training and evaluation run, the model established a strong performance baseline:

| Metric                | Score     |
| --------------------- | --------- |
| **Execution Accuracy** | **33.90%** |
| Correct Predictions   | 339 / 1000|
| Final Validation Loss | 0.071687  |

This result indicates that the `bart-base` model, after being fine-tuned on 5,000 examples, can correctly generate an executable SQL query that produces the right answer for approximately one-third of the questions in a previously unseen test set. While not perfect, this is a very solid and realistic baseline for this complex task and serves as an excellent benchmark for further improvements.

---

## Future Work & Improvements

This project has laid the groundwork for several exciting next steps that would be pursued in a real-world scenario to improve performance and usability.

1.  **Benchmark Against Other Architectures:**
    * **Action:** Fine-tune a `google/flan-t5-base` model on the same dataset and run it through the exact same execution accuracy evaluation pipeline.
    * **Goal:** Determine if a different pre-trained model architecture can outperform BART and provide a valuable comparison point for analysis.

2.  **Conduct a Detailed Error Analysis:**
    * **Action:** Systematically categorize the 661 incorrect predictions to identify common failure modes.
    * **Goal:** Answer critical questions like: Does the model struggle with `JOIN`s? Does it fail on `GROUP BY` clauses? Does it misunderstand nested queries? This analysis provides clear direction for targeted improvements.

3.  **Build and Deploy an Interactive Demo:**
    * **Action:** Use Gradio or Streamlit to create a simple web interface where a user can input a schema and a question and see the generated SQL in real-time.
    * **Goal:** Deploy this interface on Hugging Face Spaces to make the project tangible, shareable, and easily demonstrable.

---

## How to Use This Model

The fine-tuned model is publicly available on the Hugging Face Hub. You can easily load it and use it for inference.

```python
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the fine-tuned model and tokenizer from the Hub
model_id = "rkgupta3/bart-base-text-to-sql-full"
model = BartForConditionalGeneration.from_pretrained(model_id)
tokenizer = BartTokenizer.from_pretrained(model_id)

# Define your schema and question
db_schema = "CREATE TABLE artists (Artist_ID real, Artist_Name text, Age real)"
question = "How many artists are there?"

# Format the prompt exactly as in training
prompt = f"Schema: {db_schema} | Question: {question}"

# Generate the SQL query
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=128)
generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated SQL: {generated_sql}")
# Expected output: SELECT count(*) FROM artists
