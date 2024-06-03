#%%
from perplexity import Perplexity
from ../structurize/generate_json_qa import *

perplexity = Perplexity()
#%%
benchmarks = generate_bench_tree("../ControlBench/ControlBench.tex")
benchmark_dict = process_bench_tree(benchmarks)

#%%
do_section = 'Differential Equations, Laplace Transform and Preliminaries'

section_benchmarks = [val for key, val in benchmark_dict[do_section].items()]

questions = [question for question,answer in section_benchmarks]


#%%
for question in questions:
    answer = perplexity.search(question)
    for a in answer:
        print(a)
perplexity.close()