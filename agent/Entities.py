class Entity:
    def __init__(self, position, velocity, width, height):
        # Store position as a tuple
        self.position = position
        # Store velocity as a tuple
        self.velocity = velocity
        self.width = width
        self.height = height


# Player inherits from Entity
class Player(Entity):
    def __init__(self, position, velocity, width, height, **kwargs):
        # Set the attributes as per Entity
        super().__init__(position, velocity, width, height)
        for key, value in kwargs.items():
            # Set the rest of the attributes
            setattr(self, key, value)

    def isAlive(self):
        if self.player_status == 2:
            return False
        else:
            return True

    # Checks if there are bullets or enemies close. The bullets and enemies are referenced as Entities (the super class)
    def isEnemyOrHostileClose(self, entities):
        for entity in entities:
            # Consider lowering the threshold for distance
            if abs(self.position[0] - entity.position[0]) < 12 and abs(self.position[1] - entity.position[1]) < 12:
                return True

        return False


# Bullet class
class Bullet(Entity):
    def __init__(self, position, velocity, width, height):
        # Set the attributes as per Entity
        super().__init__(position, velocity, width, height)

        self.isBullet = True


# Item class
class Item(Entity):
    def __init__(self, position, velocity, width, height, **kwargs):
        # Set the attributes as per Entity
        super().__init__(position, velocity, width, height)
        self.item_type = kwargs.get('item_type', 0)


# Laser class
class Laser(Entity):
    def __init__(self, position, velocity, width, height, **kwargs):
        # Set the attributes as per Entity
        super().__init__(position, velocity, width, height)
        self.arc = kwargs.get('arc', 0)


