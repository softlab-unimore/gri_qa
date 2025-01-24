NNDemo = False
max_demo = 5
gpt_model = 'gpt-4'
program = 'sql-py'
template = 'original-sql-py-no-intermediate'


def parallel_func(i):
    max_retry = 3
    while max_retry > 0:
        try:
            codex_prompter = CodexAnswerCOTExecutor_HighTemperaturMajorityVote(
                f'prompt_template/{template}.json',
                dataset.iloc[i]['id'],
                dataset.iloc[i]['utterance'],
                dataset.iloc[i]['context'],
                dataset.iloc[i]['targetValue'],
                base_path='../dataset/WikiTableQuestions/',
                demo_file=f'few-shot-demo/WikiTQ-{program}.json',
            )
            codex_prompter.max_demo = max_demo
            codex_prompter.model = gpt_model
            codex_prompter._gen_gpt_prompt(NNDemo)
            codex_prompter._get_gpt_prediction_majority_vote(repeat_times=5)
            log = codex_prompter._log_dict()
            break
        except Exception as e:
            log = {
                'id': dataset.iloc[i]['id'],
                'uncaught_err': str(e)
            }
            if "model's maximum context length" in str(e):
                return log
            max_retry -= 1
    return log


n_threads = 3
maxLimit = 5

output_result_file = f'../dataset/WikiTableQuestions/results/CodexAnswerCOTExecutor_HighTemperaturMajorityVote_{template}_{program}_NNDemo={NNDemo}_results_pristine-unseen-tables_limit{maxLimit}_model{gpt_model}.json'
logs = Parallel(
    n_jobs=n_threads, require='sharedmem'
)(
    delayed(parallel_func)(i) for i in tqdm(range(min(maxLimit, dataset.shape[0])))
)
json.dump(logs, open(output_result_file, 'w'), indent=4)