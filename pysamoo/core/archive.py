from pymoo.core.population import Population


class Archive:

    def __init__(self, survival, max_size=100, trunc_size=None, problem=None) -> None:
        super().__init__()
        self.sols = Population()
        self.survival = survival
        self.max_size = max_size
        self.trunc_size = trunc_size if trunc_size is not None else max_size
        self.problem = problem

    def add(self, sols):

        sols = Population.merge(self.sols, sols)

        if len(sols) > self.max_size:
            sols = self.survival.do(self.problem, sols, n_survive=self.trunc_size)

        self.sols = sols

