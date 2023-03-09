import argparse
import logging
import transformers
import datasets
import sys
import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='atomic_samples.txt')
    parser.add_argument('--task', type=str, default='country_prediction')
    parser.add_argument('--max_new_tokens', type=int, default=5)
    parser.add_argument('--checkpoint', type=str, default="facebook/opt-iml-max-1.3b")
    args = parser.parse_args()

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.WARNING
    logger.setLevel(log_level)
    logging.info(args)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logging.warning("Loading Dataset.")
    folder = "data/{}/".format(args.task)

    with open(folder + args.file, 'r') as f:
        data = f.readlines()
    data = [d.replace('\n', '') for d in data]
    logging.warning("Dataset loaded")

    logging.warning("Loading model {}".format(args.checkpoint))
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if "opt" in checkpoint:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
    elif "flan" in checkpoint:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
    else:
        model = AutoModel.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

    results = []
    json_write = open("results/{}/{}_{}.json".format(args.task, args.checkpoint.split("/")[1], args.file[:-4]),
                      'w')

    for i, sample in tqdm(enumerate(data[:])):
        inputs = tokenizer.encode(sample, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs, max_new_tokens=args.max_new_tokens)
        result = {"index":i, "input":sample, "output":tokenizer.decode(outputs[0])}
        results.append(result)
        # json_write.writelines([json.dumps(r)+'\n' for r in results])
        json_write.write(json.dumps(result)+'\n')
        if i % 100 == 0:
            json_write.flush()

    json_write.close()
    logging.warning("Generation completed.")
    logging.warning("Output saved.")

if __name__ == "__main__":
    main()


