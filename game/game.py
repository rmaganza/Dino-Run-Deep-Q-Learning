from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

from utils import actions_df
from model.image_processing import grab_screen
from game.gameutils import init_script

from paths import CHROMEDRIVER_PATH
from utils import scores_df

ACTIONS = 2  # restrict scope: only jumping and doing nothing, editable to 3


class Game:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self._driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH,
                                        chrome_options=chrome_options)
        self._driver.get('https://elvisyjlin.github.io/t-rex-runner/')
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(init_script)

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def press_down(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(
            score_array)  # the javascript object is of type array with score in the format [1,0,0] which is 100.
        return int(score)

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()


class DinoAgent:
    def __init__(self, game):  # takes game as input for taking actions
        self._game = game
        self.jump()  # to start the game, we need to jump once

    def is_running(self):
        return self._game.get_playing()

    def is_crashed(self):
        return self._game.get_crashed()

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()


class GameState:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game

    def get_state(self, actions):
        actions_df.loc[len(actions_df)] = actions[1]  # storing actions in a dataframe
        score = self._game.get_score()
        reward = 0.1
        is_over = False  # game over
        if actions[1] == 1:
            self._agent.jump()
        elif ACTIONS > 2:
            if actions[2] == 1:
                self._agent.duck()
        image = grab_screen(self._game._driver)
        if self._agent.is_crashed():
            scores_df.loc[len(scores_df)] = score  # log the score when game is over
            self._game.restart()
            reward = -1
            is_over = True
            print(f"\nGAME OVER. Score: {score}\n")
        return image, reward, is_over  # return the Experience tuple
