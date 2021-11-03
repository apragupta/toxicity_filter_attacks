import subprocess
def _split_results(rows):
    """Parse the tab-delimited returned lines, modified from: https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py"""
    for line in rows:
        line = line.strip()  # remove '\n'
        if len(line) > 0:

            if line.count('\t') > 2:

                parts = line.split('\t')
                for part in parts:
                    print(part)
                tokens = parts[0].split(" ")
                tags = parts[1].split(" ")
                confidence = [float(f) for f in parts[2].split(" ")]
                _ = parts[3]

                yield tokens, tags, confidence


p = subprocess.check_output(
    'java -XX:ParallelGCThreads=2 -Xmx500m -jar ark-tweet-nlp-0.3.2.jar examples/make_examples.txt')
pos_result = p.decode('utf-8').strip('\n\n')  # get first line, remove final double carriage return
pos_result = pos_result.split('\n\n')  # split messages by double carriage returns
pos_raw_results = [pr.split('\n') for pr in pos_result]
pos_result = []
for pos_raw_result in pos_raw_results:
    pos_result.append([x for x in _split_results(pos_raw_result)])
print(pos_result)
