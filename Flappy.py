import pygame
import neat
import time
import os
import random

pygame.font.init()

WIDTH = 500
HEIGHT = 800
current_directory= os.path.dirname(os.path.abspath(__file__))

birdIMGs = [pygame.transform.scale2x(pygame.image.load(os.path.join(current_directory+"/imgs","bird1.png"))),
            pygame.transform.scale2x(pygame.image.load(os.path.join(current_directory+"/imgs","bird2.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join(current_directory+"/imgs","bird3.png")))]

pipeIMG = pygame.transform.scale2x(pygame.image.load(os.path.join(current_directory+"/imgs","pipe.png")))

BGIMG = pygame.transform.scale2x(pygame.image.load(os.path.join(current_directory+"/imgs","bg.png")))

baseIMG = pygame.transform.scale2x(pygame.image.load(os.path.join(current_directory+"/imgs","base.png")))

statFont = pygame.font.SysFont("comicsans",50)


class Bird:
    IMGs = birdIMGs
    maxRotation = 25
    rotVel = 20
    animationTime = 5

    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tickCount = 0
        self.vel = 0
        self.height = self.y
        self.imgCount = 0
        self.img = self.IMGs[0]

    def jump(self):
        self.vel = -10.5
        self.tickCount = 0
        self.height = self.y

    def move(self):
        self.tickCount += 1

        d =  self.vel*self.tickCount + 1.5*self.tickCount**2

        if d >= 16:
            d = 16
        if d < 0:
            d -= 2

        self.y = self.y+d

        if d < 0 or  self.y < self.height +50:
             if self.tilt < self.maxRotation:
                self.tilt = self.maxRotation
        elif self.tilt > -90:
                self.tilt -= self.rotVel


    def Draw(self,win):
        self.imgCount +=1

        if self.imgCount < self.animationTime:
            self.img = self.IMGs[0]
        elif self.imgCount < self.animationTime * 2:
            self.img = self.IMGs[1]
        elif self.imgCount < self.animationTime * 3:
            self.img = self.IMGs[2]
        elif self.imgCount < self.animationTime * 4:
            self.img = self.IMGs[1]
        elif self.imgCount == self.animationTime * 4 + 1:
            self.img = self.IMGs[0]
            self.imgCount = 0

        if self.tilt <= -80:
            self.img = self.IMGs[1]
            self.imgCount = self.animationTime*2

        rotateImage = pygame.transform.rotate(self.img,self.tilt)
        newReact = rotateImage.get_rect(center= self.img.get_rect(topleft = (self.x,self.y)).center)
        win.blit(rotateImage,newReact.topleft)

    def getMask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    gap = 200
    vel = 5

    def __init__(self,x):
        self.x = x
        self.height = 0
        self.gap = 200

        self.top = 0
        self.bottom = 0
        self.pipeTop = pygame.transform.flip(pipeIMG,False,True)
        self.pipeBot = pipeIMG

        self.passed = False
        self.setHeight()

    def setHeight(self):
        self.height = random.randrange(50,450)
        self.top = self.height - self.pipeTop.get_height()
        self.bot = self.height + self.gap

    def move(self):
        self.x -= self.vel

    def draw(self,win):
        win.blit(self.pipeTop, (self.x, self.top))
        win.blit(self.pipeBot, (self.x, self.bot))

    def collide(self,bird):
        birdMask = bird.getMask()
        topMask = pygame.mask.from_surface(self.pipeTop)
        botMask = pygame.mask.from_surface(self.pipeBot)

        topOffset = (self.x - bird.x, self.top - round(bird.y))
        botOffset = (self.x - bird.x, self.bot - round(bird.y))

        bPoint = birdMask.overlap(botMask,botOffset)
        tPoint = birdMask.overlap(topMask,topOffset)

        if tPoint or bPoint:
            return True

        return False

class Base:
    vel = 5
    width = baseIMG.get_width()
    img = baseIMG

    def __init__(self,y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width

    def move(self):
        self.x1 -= self.vel
        self.x2 -= self.vel

        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width
        elif self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw(self,win):
        win.blit(self.img,(self.x1,self.y))
        win.blit(self.img,(self.x2,self.y))

def drawWindow(win,birds,pipes,base,score):
    win.blit(BGIMG,(0,0))
    for pipe in pipes:
        pipe.draw(win)

    text = statFont.render("Score: " + str(score),1,(255,255,255))
    win.blit(text,(WIDTH - 10 - text.get_width(),10))

    base.draw(win)

    for bird in birds:
        bird.Draw(win)

    pygame.display.update()

def main(genomes,config):

    nets = []
    ge = []
    birds = []
    Pipe.vel = 5
    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        birds.append(Bird(230,350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(600)]
    score = 0

    win = pygame.display.set_mode((WIDTH,HEIGHT))
    clock = pygame.time.Clock()

    run = True
    while run:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit ()
                quit ()

        pipeInd = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].pipeTop.get_width() :
                pipeInd = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipeInd].height), abs(bird.y - pipes[pipeInd].bot)))

            if output[0] > 0.5:
                bird.jump()

        rem = []
        addPipe = False
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    addPipe = True

            if pipe.x + pipe.pipeTop.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if addPipe:
            score += 1
            for g in ge:
                g.fitness += 5
            Pipe.vel +=.5
            pipes.append(Pipe(700))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()
        drawWindow(win,birds,pipes,base,score)


def run(configpath):
    config = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,configpath)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,None)


if __name__ == "__main__":
    localDir = os.path.dirname(__file__)
    configPath = os.path.join(localDir,"Config.txt")
    run(configPath)










