class Container:
    def __init__(self):
        self.providers = {}

    def register(self, key, provider):
        self.providers[key] = provider

    def resolve(self, key):
        if key not in self.providers:
            raise KeyError(f"Provider '{key}' is not registered.")
        return self.providers[key]()
    
    def has(self, key):
        return key in self.providers
