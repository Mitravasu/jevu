"""Simple animated Pygame view for a JevU simulation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pygame

from jevu.agent import AgentState
from jevu.decision import ActionSelection
from jevu.game_state import GameState
from jevu.rules import FRUIT_TREE_COOLDOWN
from jevu.world import Position, TileType

BLANK_COLOR = (190, 225, 170)
TREE_COLOR = (200, 70, 70)
TREE_COOLDOWN_COLOR = (139, 69, 19)
TREE_RECOVERING_COLOR = (0, 100, 0)
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
END_BACKGROUND_COLOR = (35, 50, 40)
END_TEXT_COLOR = (240, 245, 240)
END_ACCENT_COLOR = (160, 220, 140)
CLAIM_BORDER_MIN_WIDTH = 3


def tile_color(tile: TileType, cooldown: int = 0) -> tuple[int, int, int]:
    """Return a tile color that shows fruit-tree recovery state."""

    if tile is not TileType.FRUIT_TREE:
        return BLANK_COLOR
    if cooldown >= FRUIT_TREE_COOLDOWN:
        return TREE_COOLDOWN_COLOR
    if cooldown > 0:
        return TREE_RECOVERING_COLOR
    return TREE_COLOR


def agent_color(agent_number: int) -> tuple[int, int, int]:
    """Return the stable color assigned to an agent number."""

    return AGENT_COLORS[(agent_number - 1) % len(AGENT_COLORS)]


def build_summary_lines(
    initial_state: GameState,
    final_state: GameState,
) -> tuple[str, ...]:
    """Build the game-over reason and per-agent survival summary."""

    reason = "All agents died" if not final_state.agents else "Maximum turns reached"
    final_agent_ids = {agent.id for agent in final_state.agents}
    survival_turns = {agent.id: 0 for agent in initial_state.agents}

    for entry in final_state.action_log[len(initial_state.action_log) :]:
        survival_turns[entry.agent_id] = entry.turn - initial_state.turn

    agent_lines = tuple(
        f"{agent.id} survived {survival_turns[agent.id]} "
        f"{'turn' if survival_turns[agent.id] == 1 else 'turns'}"
        + (" (alive)" if agent.id in final_agent_ids else "")
        for agent in initial_state.agents
    )
    return (reason, *agent_lines)


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
                position = Position(x, y)
                color = tile_color(
                    tile,
                    game_state.fruit_tree_cooldowns.get(position, 0),
                )
                rectangle = pygame.Rect(
                    x * self.tile_size,
                    y * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )
                pygame.draw.rect(self.screen, color, rectangle)
                pygame.draw.rect(self.screen, GRID_COLOR, rectangle, width=1)

                claimed_by = game_state.tile_claims.get(position)
                if claimed_by is not None:
                    pygame.draw.rect(
                        self.screen,
                        agent_color(claimed_by),
                        rectangle,
                        width=max(CLAIM_BORDER_MIN_WIDTH, self.tile_size // 10),
                    )

        for agent in game_state.agents:
            center = (
                agent.position.x * self.tile_size + self.tile_size // 2,
                agent.position.y * self.tile_size + self.tile_size // 2,
            )
            color = agent_color(agent.number)
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

    def draw_end_screen(
        self,
        initial_state: GameState,
        final_state: GameState,
    ) -> None:
        """Replace the grid with a game-over summary."""

        summary_lines = build_summary_lines(initial_state, final_state)
        width = max(self.screen.get_width(), 520)
        height = max(self.screen.get_height(), 180 + len(summary_lines) * 30)
        if self.screen.get_size() != (width, height):
            self.screen = pygame.display.set_mode((width, height))

        self.screen.fill(END_BACKGROUND_COLOR)
        title_font = pygame.font.Font(None, 52)
        summary_font = pygame.font.Font(None, 32)
        detail_font = pygame.font.Font(None, 26)

        title = title_font.render("Game Over", True, END_ACCENT_COLOR)
        self.screen.blit(title, title.get_rect(center=(width // 2, 55)))

        reason = summary_font.render(summary_lines[0], True, END_TEXT_COLOR)
        self.screen.blit(reason, reason.get_rect(center=(width // 2, 105)))

        for index, line in enumerate(summary_lines[1:]):
            detail = detail_font.render(line, True, END_TEXT_COLOR)
            self.screen.blit(
                detail,
                detail.get_rect(center=(width // 2, 150 + index * 28)),
            )

        footer = detail_font.render("Close the window to exit", True, END_ACCENT_COLOR)
        self.screen.blit(footer, footer.get_rect(center=(width // 2, height - 30)))
        pygame.display.flip()

    def close(self) -> None:
        self.is_open = False
        pygame.quit()


def run_pygame(
    game_state: GameState,
    max_turns: int,
    action_selector: Callable[[AgentState], ActionSelection],
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
            action_selector=action_selector,
        )
        if view.is_open:
            view.draw_end_screen(game_state, final_state)
        if hold_open and view.is_open:
            view.wait_until_closed()
        return final_state
    finally:
        view.close()
