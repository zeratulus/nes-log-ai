class System:

    components: {}

    def __init__(self, language_iso2: str):
        self.language_iso2 = language_iso2

    def add_component(self, component_name: str, component):
        self.components[component_name] = component


    def get_component(self, component_name: str):
        if self.components[component_name]:
            return self.components[component_name]

        return None