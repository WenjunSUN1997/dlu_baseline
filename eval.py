
class eval():
    def __init__(self, truth, result):
        self.truth = truth #[(label, xmin, ymin, xmax, ymax)]
        self.result = result #[(label, xmin, ymin, xmax, ymax, score)]

    def get_tpfp_i_one_pic(self):
        score =0