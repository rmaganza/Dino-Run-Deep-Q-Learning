from game.game import Game, DinoAgent, GameState
from model.model import buildmodel
from model.trainmodel import trainNetwork


# MAIN FUNCTION
def playGame(maxgames, observe=False, verbose=False):
    game = Game()
    dino = DinoAgent(game)
    game_state = GameState(dino, game)
    model = buildmodel()
    try:
        trainNetwork(model, game_state, maxgames=maxgames, observe=observe, verbose=verbose)
    except (StopIteration, KeyboardInterrupt):
        game.end()


if __name__ == '__main__':
    playGame(observe=True, maxgames=100)
