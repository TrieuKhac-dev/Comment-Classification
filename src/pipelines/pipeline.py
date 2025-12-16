from src.interfaces.pipeline import IPipeline

class Pipeline(IPipeline):
    def __init__(self, steps):
        self.steps = steps

    def run(self, context):
        for step in self.steps:
            context = step.run(context)
        return context