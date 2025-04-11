# GO-RAC

This implementation is primarily for educational purposes to explore the concepts of HNSW and vector embeddings.

Project exploring Hierarchical Navigable Small World (HNSW) graphs and vector embeddings.

## Testing the HNSW Implementation

```bash
cd testdata
```

### Required File
- `word2vec_sample.bin`: A binary file containing word embeddings in Word2Vec format. This file should be placed in the `testdata` directory.

### Obtaining Word2Vec Embeddings

Pre-trained Word2Vec embeddings can be downloaded from various sources. One common source is:

**Google's Pre-trained Word2Vec Embeddings**:
   - [Google News vectors (trained on ~3 billion words)](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
   - File size: ~1.5GB
   - Provides 300-dimensional vectors for 3 million words and phrases.

Download the `.bin` file and place it in the project's root directory initially.

## Creating a Smaller Sample Embedding File (Optional)

To create a smaller sample file (`word2vec_sample.bin`) for faster testing, run the following script from the project root. This script requires the full `GoogleNews-vectors-negative300.bin` file to be present.

```bash
./build_and_run.sh GoogleNews-vectors-negative300.bin --add_words 800
```
*This command selects 800 random words plus specific words required for the tests.*

The resulting `word2vec_sample.bin` file will be created in the `testdata` directory.

## Running the Analogy Tests

Navigate back to the project root directory and run the test script:

```bash
cd .. 
./run_analogy_tests.sh   
```

## Disclaimer

This implementation is primarily for educational purposes to explore the concepts of HNSW and vector embeddings.