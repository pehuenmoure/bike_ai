from agent import *
from bayes_opt import BayesianOptimization
from variables import *

class BayesianEvolution(object):
    def __init__(self):
        self.agent = Agent()
        self.generation = 0
        pygame.init()
        self.screen = pygame.display.set_mode((int(WIDTH), int(HEIGHT)))
        walls = []
        for i in range(WALL):
            walls.append(Wall(move = WALL_MOVE))
        self.walls = walls

    def collision(self):
        ''' Kill an agent which collides with the obstacle '''
        i = self.agent
        for j in self.walls:
            if i.pos[0] > j.pos[0] and i.pos[0] < j.pos[0] + j.width and \
            i.pos[1] < j.pos[1] + j.height and j.pos[1] < i.pos[1]:
                i.dead() # kill the agent
                i.getFitness() # evaluate the score

    def update(self, screen, draw):
        ''' every frame access the environment and agents to update the location '''
        command = self.agent.getCommand()
        self.agent.update(screen, command, draw, self.walls)
        for w in self.walls:
            w.update(screen,draw)

    def run(self, itera = 10):
        gp_params = {"alpha": 1e-5}
        bayesOpt = BayesianOptimization(self.evaluate,
            {'w0': (-1.0, 1.0),
            'w1': (-1.0, 1.0),
            'w2': (-1.0, 1.0),
            'w3': (-1.0, 1.0),
            'w4': (-1.0, 1.0),})

        bayesOpt.maximize(n_iter=itera, **gp_params,)
        print('Final Results')
        print('Bayes Optimizaiton: %f' % bayesOpt.res['max']['max_val'])  

        maxParams = bayesOpt.res['max']['max_params']
        # print(maxParams)
        while True: 
            self.evaluate(maxParams['w0'], maxParams['w1'], maxParams['w2'], maxParams['w3'], maxParams['w4'])

    def evaluate(self, w0, w1, w2, w3, w4):
        val = 0.0
        steps = 0
        self.agent.setWeights(w0, w1, w2, w3, w4)
        while self.agent.isAlive and steps <= MAX_STEPS:
            steps += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        extinct = True
            # show animation
            if (self.generation%FREQ == 0):
                self.screen.fill((0, 0, 0)) # reset canvas
                self.update(self.screen,True) # update current environment
                self.collision() # check collisions
                pygame.display.flip() # update the image
            # no animation
            else:
                self.update(self.screen,False) # update current environment
                self.collision()  # check collisions

        for w in self.walls:
            w.reset()
        self.generation += 1

        fitness = self.agent.fitness
        self.agent = Agent()

        return fitness



''' Evolution class will store all environments where the agents live and also dictate the mutation / cross over '''

class Evolution(object):
    def __init__(self, community = 10):
        ''' constructor
            [input]
            community(optional) : number of agents in each generation

            [parameter]
            agents :  array of agent object
            generation : +1 after every generation
            community : the length of agents array
            step : +1 after every frame
            screen : canvus for the animation
            target : Target object (the goal the agents try to reach)
            walls : the array of wall objects
        '''
        agents = []
        for i in range(community):
            random.seed()
            a = Agent()
            agents.append(a)
        self.agents = agents
        self.generation = 0
        self.community = community
        self.step = 0

        pygame.init() # initialize pygame environment
        self.screen = pygame.display.set_mode((int(WIDTH), int(HEIGHT)))

        self.target = Target()
        walls = []
        for i in range(WALL):
            walls.append(Wall(move = WALL_MOVE))
        self.walls = walls

        self.bestfit = np.inf # store the best agent in the simulation
        self.bestagent = []


    def update(self, screen, draw):
        ''' every frame access the environment and agents to update the location
        '''
        for i in self.agents:
            command = i.getCommand(self.step)
            # if the gene is empty the agent is dead
            if command == None:
                i.dead()
                i.getFitness(self.step) # evaluate score
            i.update(screen, command, draw, self.walls, self.step)
            for i in self.walls:
                i.update(screen,draw)
        self.step += 1

    def extinct(self):
        ''' True if all agents in currect generation is dead'''
        ans = True
        for i in self.agents:
            if i.isAlive:
                ans = False
        return ans

    def crossover(self):
        # sort the agents by the fitness
        bestChildren = sorted(self.agents, key = lambda x: x.fitness)

        nextGen = []
        for i in range(self.community):
            random.seed()
            # create the sequence gene from the 2 best parents
            Gene = []
            for j in range(GENE):
                r = random.random()
                # create a random command with given rate
                if r < MUTATION:
                    r2 = random.random()
                    if r2 < (1/3):
                        Gene.append(-1)
                    elif r2 < (2/3):
                        Gene.append(0)
                    else:
                        Gene.append(1)
                # with 50% chance from parentA, 50% chance from parentB
                elif r < (MUTATION + (1 - MUTATION) / 2):
                    Gene.append(bestChildren[0].gene[j])
                else:
                    Gene.append(bestChildren[1].gene[j])
            a = Agent()
            a.setGene(Gene)
            # create an array of agents for the next generation
            nextGen.append(a)
        return nextGen, bestChildren[0]

    def collision(self):
        ''' Kill an agent which collides with the obstacle '''
        for i in self.agents:
            for j in self.walls:
                if i.pos[0] > j.pos[0] and i.pos[0] < j.pos[0] + j.width and \
                i.pos[1] < j.pos[1] + j.height and j.pos[1] < i.pos[1]:
                    i.dead() # kill the agent
                    i.getFitness(self.step) # evaluate the score

    def hitTarget(self, step):
        ''' Kill an agent which reach the destination '''
        for i in self.agents:
                if i.pos[0] > self.target.pos[0]-10 and i.pos[0] < self.target.pos[0] + 10 and \
                i.pos[1] < self.target.pos[1]+10 and self.target.pos[1]-10 <  i.pos[1]:
                    i.dead() # kill the agent
                    i.fitness = -(1000-step) # evaluate the score ( is the best)


    def run(self):
        ''' Run the simulation '''
        done = True
        #clock = pygame.time.Clock()
        extinct = False
        # if all agents in the environment died, start new generation
        while not extinct:
            # quit when the window is closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        extinct = True
            # show animation
            if (self.generation%FREQ == 0):
                self.screen.fill((0, 0, 0)) # reset canvas
                self.target.draw(self.screen) # draw target
                self.update(self.screen,True) # update current environment
                self.collision() # check collisions
                self.hitTarget(self.step) # check if the agent reached the destination
                pygame.display.flip() # update the image
            # no animation
            else:
                self.update(self.screen,False) # update current environment
                self.collision()  # check collisions
                self.hitTarget(self.step) # check if the agent reached the destination


                # pressed = pygame.key.get_pressed()
                # if event.type == pygame.KEYDOWN and pressed[pygame.K_LEFT]:
                #     agent.update(screen,-1)
                # elif event.type == pygame.KEYDOWN and pressed[pygame.K_RIGHT]:
                #     agent.update(screen,1)
                # else:
                #     agent.update(screen,0)

                #clock.tick(60)

            extinct = self.extinct() # check extinction

        self.generation += 1 # +1 to the generation
        self.step = 0 # reset frame

        self.agents, best = self.crossover() # create new generation
        # update best so far
        if best.fitness < self.bestfit:
            self.bestfit = best.fitness
            self.bestagent = [best]
        # append the agent with the best fitness so far to the next generation
        if len(self.bestagent)>0:
            a = Agent()
            a.setGene(self.bestagent[0].gene)
            a.setElite() # color differently
            self.agents.pop()
            self.agents.append(a)
        # print out update
        # if (self.generation % 10 == 0):
        print('{} generation: {} | {} '.format(self.generation, best.fitness, self.bestfit))
        for i in self.walls:
            i.reset()
        self.run() # start new generation
