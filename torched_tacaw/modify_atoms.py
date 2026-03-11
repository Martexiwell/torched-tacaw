import ase

class RecipeBuilder:
    def __init__(self, atoms:ase.Atoms):
        self.atoms = atoms
        self.operations = []

    def rotate(self, angle):
        # Assuming you have a method to rotate atoms
        self.atoms.rotate(angle)
        self.operations.append({'rotate': angle})

    def translate(self, vector):
        self.atoms.translate(vector)
        self.operations.append({'translate': vector})

    def get_operations(self):
        return self.operations


class AtomModifier(RecipeBuilder):
    def __init__(
            self,
            atoms: ase.Atoms,
            recipe: list[dict]
    ):
        super().__init__(atoms)
        self.recipe:list[dict] = recipe

        for step in self.recipe:
            ...