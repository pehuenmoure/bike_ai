from agent import *
import matplotlib.pyplot as plt

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
            a = Agent(pos = np.array([50., HEIGHT - 50.]))
            agents.append(a)
        self.agents = agents
        self.generation = 0
        self.community = community
        self.step = 0

        # pygame.init() # initialize pygame environment
        # self.screen = pygame.display.set_mode((int(WIDTH), int(HEIGHT)))
        self.screen = None
        self.target = Target()
        walls = []
        for i in range(WALL):
            walls.append(Wall(move = WALL_MOVE))
        self.walls = walls

        food = []
        for i in range(FOOD):
            food.append(Food())
        self.food = food

        self.bestfit = np.inf # store the best agent in the simulation
        self.bestagent = []

    def update(self, screen, draw):
        ''' every frame access the environment and agents to update the location
        '''
        for i in self.agents:
            command = i.getCommand(self.step)
            # if the gene is empty the agent is dead
            if command == None:
                i.getFitness(self.step) # evaluate score
                i.dead()
            i.update(screen, command, draw, self.step, self.walls)
            for i in self.walls:
                i.update(screen,draw)
        if draw:
            for i in self.food:
                i.draw(screen)
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
            a = Agent(pos = np.array([50., HEIGHT - 50.]))
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
                    i.getFitness(self.step) # evaluate the score
                    i.dead()

    def hitTarget(self, step):
        ''' Kill an agent which reach the destination '''
        for i in self.agents:
                if i.pos[0] > self.target.pos[0]-10 and i.pos[0] < self.target.pos[0] + 10 and \
                i.pos[1] < self.target.pos[1]+10 and self.target.pos[1]-10 <  i.pos[1]:

                    i.fitness = -(GENE-step) # evaluate the score ( is the best)
                    i.dead() # kill the agent

    def recoverHealth(self):
        for i in self.agents:
            for f in self.food:
                dist = np.linalg.norm(i.pos - f.pos)
                if dist < 5:
                    # f.color = (125,125,125)
                    i.health += NUTRIENT

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
                self.recoverHealth() # check food
                self.hitTarget(self.step) # check if the agent reached the destination
                pygame.display.flip() # update the image
            # no animation
            else:
                self.update(self.screen,False) # update current environment
                self.collision()  # check collisions
                self.recoverHealth() # check food
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
            a = Agent(pos = np.array([50., HEIGHT - 50.]))
            a.setGene(self.bestagent[0].gene)
            a.setElite() # color differently
            self.agents.pop()
            self.agents.append(a)
        # print out update
        if (self.generation % 10 == 0):
            print('{} generation: {} | {} '.format(self.generation, best.fitness, self.bestfit))
        for i in self.walls:
            i.reset()
        self.run() # start new generation

class DeepEvolution(Evolution):
    def __init__(self, comm = 10, train_iteration = 1000):
        super().__init__(community = comm)
        # competition of 2 agents
        c_agents = []
        for i in range(comm):
            random.seed()
            a = Agent(deep = True, pos = np.array([50., HEIGHT - 50.]), mode = 'chaser')
            c_agents.append(a)
        self.chaser = c_agents

        r_agents = []
        for i in range(comm):
            random.seed()
            a = Agent(deep = True, pos = np.array([WIDTH - 50., 50.]), direction = np.array([0.,1.]), mode = 'chaser')
            r_agents.append(a)
        self.runner = r_agents

        self.communityNum = 0;

    def update(self, screen, draw):
        ''' every frame access the environment and agents to update the location
        '''
        for i in self.walls:
            i.update(screen,draw)

        self.runner[self.communityNum].sensor.update(self.runner[self.communityNum].pos, self.runner[self.communityNum].dir, self.walls)
        self.chaser[self.communityNum].sensor.update(self.chaser[self.communityNum].pos, self.chaser[self.communityNum].dir, self.walls)

        commandR = self.runner[self.communityNum].nn.predict(np.array(self.runner[self.communityNum].sensor.vector).reshape(1,7))
        commandC = self.chaser[self.communityNum].nn.predict(np.array(self.chaser[self.communityNum].sensor.vector).reshape(1,7))
        move = [-1,0,1]
        self.runner[self.communityNum].update(screen, commandR, draw, self.step, opponent = self.chaser[self.communityNum])
        self.chaser[self.communityNum].update(screen, commandC, draw, self.step, opponent = self.runner[self.communityNum])

        if draw:
            for i in self.food:
                i.draw(screen)
        self.step += 1

    def collision(self):
        ''' Kill an agent which collides with the obstacle '''
        i = self.runner[self.communityNum]
        for j in self.walls:
            if i.pos[0] > j.pos[0] and i.pos[0] < j.pos[0] + j.width and \
            i.pos[1] < j.pos[1] + j.height and j.pos[1] < i.pos[1]:
                if i.isAlive:
                    i.getFitnessChaser(self.step, self.chaser[self.communityNum]) # evaluate the score
                i.dead() # kill the agent
        i = self.chaser[self.communityNum]
        for j in self.walls:
            if i.pos[0] > j.pos[0] and i.pos[0] < j.pos[0] + j.width and \
            i.pos[1] < j.pos[1] + j.height and j.pos[1] < i.pos[1]:
                if i.isAlive:
                    i.getFitnessChaser(self.step, self.runner[self.communityNum]) # evaluate the score
                i.dead() # kill the agent
    def check_gameover(self):
        return not self.runner[self.communityNum].isAlive and not self.chaser[self.communityNum].isAlive

    def AgentsCollision(self, step):
        if self.runner[self.communityNum].pos[0] == self.chaser[self.communityNum].pos[0] and \
        self.runner[self.communityNum].pos[1] == self.chaser[self.communityNum].pos[1]:
            self.runner[self.communityNum].dead()
            self.chaser[self.communityNum].dead()

            self.runner[self.communityNum].getFitnessChaser(step, self.chaser[self.communityNum])

            self.chaser[self.communityNum].getFitnessChaser(step, self.runner[self.communityNum])

    def NNcrossOver(self, A, B):
        ''' create W and B '''
        random.seed()

        wA = A.nn.getW()
        bA = A.nn.getB()

        wB = B.nn.getW()
        bB = B.nn.getB()

        w = {}
        matrix = []
        for i in range(n_input):
            vector = []
            for j in range(n_hidden_1):
                r = random.random()
                # create a random command with given rate
                if r < MUTATION:
                    vector.append(random.random())
                # with 50% chance from parentA, 50% chance from parentB
                elif r < (MUTATION + (1 - MUTATION) / 2):
                    vector.append(wA['h1'].item(i,j))
                else:
                    vector.append(wB['h1'].item(i,j))
            matrix.append(vector)
        w['h1'] = np.matrix(matrix)

        matrix = []
        for i in range(n_hidden_1):
            vector = []
            for j in range(n_hidden_2):
                r = random.random()
                # create a random command with given rate
                if r < MUTATION:
                    vector.append(random.random())
                # with 50% chance from parentA, 50% chance from parentB
                elif r < (MUTATION + (1 - MUTATION) / 2):
                    vector.append(wA['h2'].item(i,j))
                else:
                    vector.append(wB['h2'].item(i,j))
            matrix.append(vector)
        w['h2'] = np.matrix(matrix)

        matrix = []
        for i in range(n_hidden_2):
            vector = []
            for j in range(n_classes):
                r = random.random()
                # create a random command with given rate
                if r < MUTATION:
                    vector.append(random.random())
                # with 50% chance from parentA, 50% chance from parentB
                elif r < (MUTATION + (1 - MUTATION) / 2):
                    vector.append(wA['out'].item(i,j))
                else:
                    vector.append(wB['out'].item(i,j))
            matrix.append(vector)
        w['out'] = np.matrix(matrix)

        b = {}
        vector = []
        for i in range(n_hidden_1):
            r = random.random()
            # create a random command with given rate
            if r < MUTATION:
                vector.append(random.random())
            # with 50% chance from parentA, 50% chance from parentB
            elif r < (MUTATION + (1 - MUTATION) / 2):
                vector.append(bA['b1'].item(i))
            else:
                vector.append(bB['b1'].item(i))
        b['b1'] = np.array(vector)

        vector = []
        for i in range(n_hidden_2):
            r = random.random()
            # create a random command with given rate
            if r < MUTATION:
                vector.append(random.random())
            # with 50% chance from parentA, 50% chance from parentB
            elif r < (MUTATION + (1 - MUTATION) / 2):
                vector.append(bA['b2'].item(i))
            else:
                vector.append(bB['b2'].item(i))
        b['b2'] = np.array(vector)

        vector = []
        for i in range(n_hidden_2):
            r = random.random()
            # create a random command with given rate
            if r < MUTATION:
                vector.append(random.random())
            # with 50% chance from parentA, 50% chance from parentB
            elif r < (MUTATION + (1 - MUTATION) / 2):
                vector.append(bA['out'].item(i))
            else:
                vector.append(bB['out'].item(i))
        b['out'] = np.array(vector)

        return w, b

    def train(self):
        runner_status = []
        chaser_status = []
        print('initiate Training')
        for i in range(1000):
            ''' train 2 agents '''
            runner_store = []
            chaser_store = []
            for i in range(self.community):
                GAME_OVER = False
                while not GAME_OVER:
                    # for event in pygame.event.get():
                    #     if event.type == pygame.QUIT:
                    #             extinct = True
                    # self.screen.fill((0, 0, 0)) # reset canvas
                    self.update(self.screen, False)
                    self.collision()
                    self.AgentsCollision(self.step)  # check collisions
                    GAME_OVER = self.check_gameover()
                    # pygame.display.flip() # update the image
                runner_store.append((self.runner[self.communityNum].fitness, self.communityNum))
                chaser_store.append((self.chaser[self.communityNum].fitness, self.communityNum))
                self.communityNum += 1
            print('End of generation')
            self.communityNum = 0
            walls = []
            for i in range(WALL):
                walls.append(Wall(move = WALL_MOVE))
            self.walls = walls

            runner_store.sort()
            chaser_store.sort()
            runner_status.append(runner_store[0][0])
            parentA_runner = runner_store[0][1]
            parentB_runner = runner_store[1][1]

            chaser_status.append(chaser_store[0][0])
            chaserA_runner = chaser_store[0][1]
            chaserB_runner = chaser_store[1][1]

            c_agents = []
            for i in range(self.community):
                random.seed()
                a = Agent(deep = True, pos = np.array([50., HEIGHT - 50.]), mode = 'chaser')
                w,b = self.NNcrossOver(self.chaser[chaserA_runner], self.chaser[chaserB_runner])
                a.nn.setB(b)
                a.nn.setW(w)
                c_agents.append(a)
            self.chaser = c_agents

            r_agents = []
            for i in range(self.community):
                random.seed()
                a = Agent(deep = True, pos = np.array([WIDTH - 50., 50.]), direction = np.array([0.,1.]), mode = 'chaser')
                w,b = self.NNcrossOver(self.runner[parentA_runner], self.runner[parentB_runner])
                a.nn.setB(b)
                a.nn.setW(w)
                r_agents.append(a)
            self.runner = r_agents


        print('Terminate Training')
        fig = plt.figure()
        plt.title('Training Progress')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.plot(runner_status)
        plt.plot(chaser_status)
        plt.legend(['Chaser1', 'Chaser2'])
        fig.savefig('training.png')

if __name__ == '__main__':
    d = DeepEvolution(comm = 10)
    d.train()
