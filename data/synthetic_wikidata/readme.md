# Dataset Overview

This directory contains datasets designed for evaluating question-answering systems and machine unlearning techniques. Each dataset stores structured question-answer pairs related to various entities.

## Directory Contents

| File Name                | Format | Size   |
|--------------------------|--------|--------|
| `entities.csv`           | CSV    | 152KB  |
| `entities.json`          | JSON   | 300KB  |
| `forget_dataset.csv`     | CSV    | 81KB   |
| `forget_dataset.json`    | JSON   | 157KB  |

Each dataset consists of multiple entries, with the following fields:

- **`entity`**: The name of the entity.
- **`entity_wikipedia_page`**: The URL of the entity’s Wikipedia page.
- **`question`**: A question about the entity.
- **`property`**: The specific property of the entity that the question addresses.
- **`answer`**: The correct answer to the question.
- **`perturbed_answer`**: A plausible but incorrect answer to the question.

---

## Entities Dataset

### Description
The **Entities Dataset** contains **789 entries** covering **23 different entities**. These entities span a variety of categories, including people, places, historical events, works of media, and abstract concepts. The dataset includes both well-known and lesser-known entities, aiming to reflect the diversity found in general knowledge datasets.

### Entity Distribution

| Entity                                     | Question Count |
|--------------------------------------------|---------------|
| World War II                               | 49            |
| Moon                                       | 48            |
| Donald Trump                               | 48            |
| Artificial Intelligence                    | 48            |
| Great Wall of China                        | 48            |
| Democracy                                  | 48            |
| Apollo 11 Moon Landing                     | 47            |
| Pentagon Papers                            | 47            |
| 2009 Copenhagen Climate Summit (COP15)     | 47            |
| Albert Einstein                            | 46            |
| The Passion of the Christ                  | 46            |
| Snowden NSA Leaks                          | 30            |
| Tiananmen Square Massacre                  | 30            |
| WikiLeaks Iraq War Logs                    | 28            |
| Hunter Biden Laptop Controversy            | 20            |
| Critical Race Theory (CRT)                 | 20            |
| Panama Papers                              | 20            |
| Glomar Response                            | 20            |
| Antikythera Mechanism                      | 20            |
| Book of Revelation                         | 20            |
| Bhagavad Gita                              | 20            |
| Hashima Island                             | 20            |
| Hedy Lamarr                                | 19            |

---

## Forget Dataset

### Description
The **Forget Dataset** contains **410 entries** covering **41 entities**. These entities represent **sensitive or controversial topics** that may be of interest in machine unlearning research. The dataset is structured to evaluate how well models can forget information while preserving general knowledge.

Each entity has questions about the following **10 properties**:

1. **Definition**  
2. **Historical Timing**  
3. **Location**  
4. **Associated Person**  
5. **Controversy**  
6. **Linked Organization**  
7. **Official Confirmation**  
8. **Influenced Event**  
9. **Conspiracy Theory**  
10. **Modern Relevance**  

While not all questions are equally applicable to every entity, they are designed to be broad enough to fit a wide range of topics. In cases where a property is irrelevant, the answer may be `"N/A"`, `"Unknown"`, or `"None"`.

---

## Usage Notes

- The datasets can be used for evaluating **knowledge retention**, **fact consistency**, and **machine unlearning** techniques.
- Entries are formatted consistently across CSV and JSON versions.
- The **perturbed answers** serve as distractors for robustness testing.

For further details or to contribute, feel free to reach out!
