"""Simple animated Pygame view for a JevU simulation."""

from __future__ import annotations

from pathlib import Path

import pygame

from jevu.game_state import GameState
from jevu.world import TileType

BLANK_COLOR = (190, 225, 170)
TREE_COLOR = (200, 70, 70)
GRID_COLOR = (80, 100, 75)
TEXT_COLOR = (20, 20, 20)
AGENT_COLORS = (
    (65, 105, 225),
    (255, 165, 0),
    (148, 0, 211),
    (0, 180, 180),
    (255, 105, 180),
    (255, 215, 0),
    (46, 139, 87),
    (139, 69, 19),
    (100, 149, 237),
    (220, 20, 60),
    (127, 255, 0),
    (138, 43, 226),
)


class PygameView:
    """Draw successive game states in a Pygame window."""

    def __init__(self, game_state: GameState, turns_per_second: float = 2.0) -> None:
        if turns_per_second <= 0:
            raise ValueError("Turns per second must be positive")

        pygame.init()
        largest_dimension = max(
            game_state.world.config.width,
            game_state.world.config.height,
        )
        self.tile_size = max(16, min(64, 800 // largest_dimension))
        self.screen = pygame.display.set_mode(
            (
                game_state.world.config.width * self.tile_size,
                game_state.world.config.height * self.tile_size,
            )
        )
        pygame.display.set_caption("JevU")
        self.font = pygame.font.Font(None, max(14, self.tile_size // 2))
        self.clock = pygame.time.Clock()
        self.turns_per_second = turns_per_second
        self.is_open = True

    def draw(self, game_state: GameState) -> bool:
        """Draw one state and return whether the window remains open."""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_open = False
                return False

        for y, row in enumerate(game_state.world.tiles):
            for x, tile in enumerate(row):
                color = TREE_COLOR if tile is TileType.FRUIT_TREE else BLANK_COLOR
                rectangle = pygame.Rect(
                    x * self.tile_size,
                    y * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )
                pygame.draw.rect(self.screen, color, rectangle)
                pygame.draw.rect(self.screen, GRID_COLOR, rectangle, width=1)

        for agent in game_state.agents:
            center = (
                agent.position.x * self.tile_size + self.tile_size // 2,
                agent.position.y * self.tile_size + self.tile_size // 2,
            )
            color = AGENT_COLORS[(agent.number - 1) % len(AGENT_COLORS)]
            pygame.draw.circle(self.screen, color, center, self.tile_size // 3)
            label = self.font.render(agent.id, True, TEXT_COLOR)
            self.screen.blit(label, label.get_rect(center=center))

        pygame.display.flip()
        self.clock.tick(self.turns_per_second)
        return True

    def wait_until_closed(self) -> None:
        """Keep the final state visible until the window is closed."""

        while self.is_open:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_open = False
            self.clock.tick(30)

    def close(self) -> None:
        self.is_open = False
        pygame.quit()


def run_pygame(
    game_state: GameState,
    max_turns: int,
    turns_per_second: float = 2.0,
    log_directory: str | Path | None = "logs",
    hold_open: bool = True,
) -> GameState:
    """Animate a game simulation and return its final state."""

    view = PygameView(game_state, turns_per_second=turns_per_second)
    try:
        final_state = game_state.run(
            max_turns=max_turns,
            log_directory=log_directory,
            state_callback=view.draw,
        )
        if hold_open and view.is_open:
            view.wait_until_closed()
        return final_state
    finally:
        view.close()
