import pygame
import time
import random
pygame.font.init()


HEIGHT, WIDTH = 800, 1000
WIN = pygame.display.set_mode((WIDTH,HEIGHT))
BG = pygame.transform.scale(pygame.image.load("space.jpg"), (WIDTH, HEIGHT))
pygame.display.set_caption("Space Dodger")

SPD = 5
FONT = pygame.font.SysFont("comicsans",30)
STAR_WIDTH, STAR_HEIGHT = 10, 10
STAR_SPD = 3

POWERUP_WIDTH, POWERUP_HEIGHT = 30, 30
POWERUP_SPD = 5
POWERUP_SPAWN_TIME = 8000
POWERUP_DURATION = 5000
POWERUP_TYPES = {
    "speed": {"color": "blue", "effect": "speed"},
    "resize": {"color": "green", "effect": "resize"},
    "invincibility": {"color": "yellow", "effect": "invincibility"}, "dodge": {"color": "pink","effect":"dodge"}, "ui": {"color": "orange", "effect": "ui"}}

def draw(player, elapsed, stars, powerups, difficulty, overlay):
    WIN.blit(BG, (0,0))
    time_text = FONT.render(f"Time: {round(elapsed)}s", 1, "white")
    diff = FONT.render(f"Difficulty: {difficulty}", 1, "purple")
    WIN.blit(diff, (WIDTH-10-diff.get_width(),10))
    WIN.blit(time_text, (10, 10))
    pygame.draw.rect(WIN, "red", player)
    WIN.blit(overlay, (player.x, player.y))
    for star in stars:
        pygame.draw.rect(WIN, "white", star)
    for powerup in powerups:
        pygame.draw.rect(WIN, powerup["color"], powerup["rect"])
    pygame.display.update()

PW, PH = 35, 50

def main():
    global STAR_SPD, SPD, STAR_HEIGHT, STAR_WIDTH, POWERUP_SPAWN_TIME, POWERUP_SPD
    run = True
    pw, ph = PW//2, PH//2
    player = pygame.Rect(WIDTH//2-PW, HEIGHT - PH, PW, PH)
    overlay = pygame.image.load("spaceship.png")
    scaled_overlay = pygame.transform.scale(overlay, (PW, PH))
    clock = pygame.time.Clock()
    start_time, elapsed = time.time(), 0
    last_powerup_spawn_time = pygame.time.get_ticks()
    active_powerup = None
    powerup_end_time = 0
    original_speed = SPD
    original_size = (PW, PH)
    star_add_increment = 500
    star_count = 0
    stars = []
    powerups = []
    invincibility = False
    ultra_instinct = False
    hit = False
    ui = False
    difficulty = 0

    while run:
        star_count += clock.tick(60)
        elapsed = time.time() - start_time
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        diffcheck = elapsed - 10*difficulty
        if diffcheck >= 10:
            STAR_SPD += 1
            star_add_increment *= 0.9
            difficulty += 1
            if not difficulty % 2:
                SPD += 1
            if not difficulty % 5:
                STAR_HEIGHT += 1
                STAR_WIDTH += 1
            if not difficulty % 3:
                star_add_increment = max(50, star_add_increment-30)
                POWERUP_SPD += 1

        if star_count > star_add_increment:
            for _ in range(difficulty//5+1):
                star_x = random.randint(0, WIDTH-STAR_WIDTH)
                star = pygame.Rect(star_x, -STAR_HEIGHT, STAR_WIDTH, STAR_HEIGHT)
                stars.append(star)
                star_count = 0

        if active_powerup and pygame.time.get_ticks() > powerup_end_time:
            if active_powerup["effect"] == "speed": SPD = original_speed
            elif active_powerup["effect"] == "resize": 
                player.width, player.height = original_size
                player.y = player.y - PH//2 if player.y < 750 else HEIGHT-PH
                scaled_overlay = pygame.transform.scale(overlay, (PW, PH))
            elif active_powerup["effect"] == "invincibility": invincibility = False
            elif active_powerup["effect"] == "ui": ui = False
            else:
                ultra_instinct = False
            active_powerup = None
        current_time = pygame.time.get_ticks()

        if current_time - last_powerup_spawn_time > POWERUP_SPAWN_TIME:
            powerup_type = random.choice(list(POWERUP_TYPES.values()))
            star = pygame.Rect(star_x, -STAR_HEIGHT, STAR_WIDTH, STAR_HEIGHT)
            powerup_x = random.randint(0, WIDTH - POWERUP_WIDTH)
            powerup_y = random.randint(0, HEIGHT // 2)
            powerup_rect = pygame.Rect(powerup_x, powerup_y, POWERUP_WIDTH, POWERUP_HEIGHT)
            powerups.append({'rect': powerup_rect, 'color': powerup_type['color'], 'effect': powerup_type['effect']})
            last_powerup_spawn_time = current_time

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and player.x - SPD >= 0:
            player.x -= SPD
        if keys[pygame.K_RIGHT] and player.x + SPD + PW <= WIDTH:
            player.x += SPD
        if keys[pygame.K_UP] and player.y - SPD > 0:
            player.y -= SPD
        if keys[pygame.K_DOWN] and player.y + SPD + PH <= HEIGHT:
            player.y += SPD
        for star in stars[:]:
            star.y += STAR_SPD if not ui else STAR_SPD//2
            if star.y > HEIGHT:
                stars.remove(star)
            elif star.y + star.height >= player.y and star.colliderect(player):
                if not ultra_instinct:
                    stars.remove(star)
                    if not invincibility:
                        hit = True
                        break
                else:
                    if player.x < HEIGHT//2: player.x = player.x + STAR_WIDTH
                    else: player.x = player.x - STAR_WIDTH

        for powerup in powerups[:]:
            powerup["rect"].y += POWERUP_SPD
            if powerup["rect"].y > HEIGHT:
                powerups.remove(powerup)
            elif powerup["rect"].y + POWERUP_HEIGHT >= powerup["rect"].y and player.colliderect(powerup['rect']):
                active_powerup = powerup
                powerup_end_time = pygame.time.get_ticks() + POWERUP_DURATION
                powerups.remove(powerup)
                if powerup['effect'] == "speed":
                    SPD *= 2
                elif powerup['effect'] == "resize":
                    player.y = player.y + ph if player.y < 750 else HEIGHT - ph
                    player.height = ph
                    player.width = pw
                    scaled_overlay = pygame.transform.scale(overlay, (pw, ph))
                elif powerup['effect'] == "invincibility":
                    invincibility = True
                elif powerup["effect"] == "dodge":
                    ultra_instinct = True
                elif powerup["effect"] == "ui":
                    ui = True
        if hit:
            pygame.display.update()
            lost_text = FONT.render(f"You lost! Total time: {round(elapsed)}s", 1, "white")
            WIN.blit(lost_text, (WIDTH//2 - lost_text.get_width()//2, HEIGHT//2 - lost_text.get_height()//2))
            pygame.display.update()
            print(f"Final time: {round(elapsed)}s")
            pygame.time.delay(2000)
            break
        draw(player, elapsed, stars, powerups, difficulty, scaled_overlay)
    pygame.quit()

if __name__ == "__main__":
    main()
