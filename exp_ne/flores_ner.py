from datasets import load_dataset, get_dataset_config_names
import openai
from sacrebleu import BLEU, CHRF
import dotenv
import argparse
from tqdm import tqdm
import json
import os
from typing import List, Dict

dotenv.load_dotenv()
client = openai.OpenAI()


def ner(model: str,
        user_prompt: str,
        system_prompt: str) -> str:
    """Run NER."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    return completion.choices[0].message.content


def evaluate_metrics(refs: List[str],
                     hyps: List[str],
                     cheat_brevity_penalty: bool = False) -> Dict[str, float]:
    """Evaluate the results with BLEU and ChrF++."""
    if cheat_brevity_penalty:
        # Add dummy words to the hypotheses to avoid brevity penalty
        # This is a hack to avoid the brevity penalty in BLEU
        hyps = [" ".join(hyp.split() + ["dummy"] * 50) for hyp in hyps]
    bleu = BLEU().corpus_score(hyps, [refs])
    chrf = CHRF(word_order=2).corpus_score(hyps, [refs])
    return {"bleu": bleu.score,
            "chrf": chrf.score}
    

langs = [name for name in get_dataset_config_names("openlanguagedata/flores_plus")
         if "Latn" in name]


def evaluate_all(langs: List[str],
                 hyps: List[str],
                 cheat_brevity_penalty: bool = False) -> Dict[str, Dict[str, float]]:
    """Evaluate it on all languages."""
    results = {}
    for lang in langs:
        print(f"Evaluating {lang}...")
        # Load the dataset
        try:
            ds = load_dataset("openlanguagedata/flores_plus",
                            lang,
                            split="dev")
        except ValueError as e: # no dev
            print(f"No dev set found for {lang}: {e}")
            continue
        refs = [ds[i]["text"] for i in range(len(ds))]
        scores = evaluate_metrics(refs, hyps, cheat_brevity_penalty)
        print(f"Scores for {lang}:")
        print("BLEU:", scores["bleu"])
        print("ChrF++:", scores["chrf"])
        # Store the results
        results.update({lang: scores})
    return results


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser(description="Run NER and save the results.")
    parser.add_argument(
        "--user_prompt_template_file",
        type=str,
        default="user_prompt_template.txt",
        help="User prompt template file."
    )
    parser.add_argument(
        "--system_prompt_template_file",
        type=str,
        default="system_prompt.txt",
        help="System prompt template file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="results.json",
        help="File to save the results."
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="eval.json",
        help="File to save the evaluation results."
    )
    parser.add_argument(
        "--cheat_brevity_penalty",
        action="store_true",
        help="Cheat the brevity penalty."
    )
    args = parser.parse_args()
    return args


def main():
    """Main function."""
    args = get_args()
    
    if not os.path.exists(args.results_file):
        # Load the dataset
        english = load_dataset("openlanguagedata/flores_plus",
                            "eng_Latn",
                            split="dev")
        sents = [english[i]["text"] for i in range(len(english))]
        results = []
        
        with open(args.user_prompt_template_file, "r") as f:
            user_prompt_template = f.read()
        with open(args.system_prompt_template_file, "r") as f:
            system_prompt_template = f.read()
            
        print("Running NER...")
            
        for sent in tqdm(sents):
            user_prompt = user_prompt_template.format(text=sent)
            system_prompt = system_prompt_template
            result = ner(args.model, user_prompt, system_prompt)
            results.append(result)
        
        print("NER completed.")
        
        eng_ne = {"eng_Latn": sents,
                "named_entities": results}
        
        # Save the results
        print("Saving results...")
        with open(args.results_file, "w") as f:
            json.dump(eng_ne, f)
        print("Results saved to", args.results_file)
    
    else:
        print("Results already exist. Loading from", args.results_file)
        with open(args.results_file, "r") as f:
            eng_ne = json.load(f)
            
    eng = eng_ne["eng_Latn"]
    ne = eng_ne["named_entities"]
    
    print("Running evaluation for all languages...)}")
    results = evaluate_all(langs, ne, args.cheat_brevity_penalty)
    print("Evaluation completed.")
    
    # Save the evaluation results
    with open(args.eval_file, "w") as f:
        json.dump(results, f)
    print("Evaluation results saved to", args.eval_file)
    print("Evaluation completed.")


if __name__ == "__main__":
    main()