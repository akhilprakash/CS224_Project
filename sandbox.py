"""
Temporary holding place for code
"""
def help0DegreeArticles(self, iterations, users, N=4):
    if iterations % N == 0:
        articles = self.network.getArticlesWithDegree0()
        for a in articles:
            for u in users:
                probLike = self.pLike(u, a)
                if random.random() < probLike:
                    self.network.addEdge(u, a)

    def help0DegreeUsers(self, iterations, article, N=5):
        if iterations % N == 0:
            users = self.network.getUsersWithDegree0()
            for u in users:
                probLike = self.pLike(u, article)
                if random.random() < probLike:
                    self.network.addEdge(u, article)

def forceConnectedGraph(self, iterations, article):
    if iterations == 0:
        readers = self.network.users.values()
        for reader in readers:
            self.network.addEdge(reader, article)


    def triadicClosureBasedOnFriends(self, iterations):
        article = self.introduceArticle(iterations)
        randReaders = random.sample(self.network.users.keys(), 1)
        for reader in randReaders:
            probLike = self.pLike(self.network.users[reader], article)
            rand = random.random()
            if rand < probLike:
                self.network.addEdge(self.network.users[reader], article)
                neighbors = self.network.friendGraph.GetNI(reader).GetOutEdges()
                neighs = []
                for n in neighbors:
                    neighs.append(n)
                randNeighbor = random.sample(neighs, 1)
                #can either force neighrbot to read it ord test with pLike
                if self.force:
                    self.network.addEdge(self.network.getUser(randNeighbor[0]), article)
                else:
                    if self.pLike(self.network.getUser(randNeighbor[0]), article) < random.random():
                        self.network.addEdge(self.network.getUser(randNeighbor[0]), article)
        readers = self.network.users.values()
        self.runRecommendation(readers)
        if self.shouldHelp0DegreeUsers:
            self.help0DegreeUsers(iterations, article)
        if self.shouldHelp0DegreeArticles:
            self.help0DegreeArticles(iterations, self.network.users.values())
        self.runAnalysis(iterations)

    def randomRandomCompleteTriangles(self, iterations):
        article = self.introduceArticle(iterations)
        randReaders = random.sample(self.network.users.keys(), 1)
        for reader in randReaders:
            probLike = self.pLike(self.network.users[reader], article)
            rand = random.random()
            if rand < probLike:
                self.network.addEdge(self.network.users[reader], article)
                neighbors = self.network.userArticleGraph.GetNI(reader).GetOutEdges()
                neighs = []
                for n in neighbors:
                    neighs.append(n)
                rand = random.sample(neighs, 1)
                print rand
                #rand is an article
                users = []
                for r in rand:
                    neighbors = self.network.userArticleGraph.GetNI(r).GetOutEdges()
                    for n in neighbors:
                        users.append(n)
                rand = random.sample(users, 1)
                for r in rand:
                    self.network.addEdge(self.network.users[r], article)
        self.runRecommendation(randReader)
        if self.shouldHelp0DegreeUsers:
            self.help0DegreeUsers(iterations, article)
        if self.shouldHelp0DegreeArticles:
            self.help0DegreeArticles(iterations, self.network.users.values())
        self.runAnalysis(iterations)

    def simulate(self, iterations):
        readers = self.network.getNextReaders() # get readers that can read at this time point

        # Introduce a new article
        article = self.introduceArticle(iterations)

        for reader in readers: # ask each reader if like it or not
            if random.random() < self.pLike(reader, article):
                self.network.addEdge(reader, article)

        self.runRecommendation(readers)
        if self.popular:
            # On every third iteration, "show" the readers the top 5 most popular articles
            if iterations % 3 == 0:
                articleDeg = evaluation.getArticleDegreeDistribution(self.network, 'alive')
                sortedDeg = sorted(articleDeg, key=lambda x: x[1], reverse=True)
                topFive = sortedDeg[0:5]
                for (aId, _) in topFive:
                    article = self.network.getArticle(aId)
                    for reader in readers:
                        probLike = plike(reader, article)
                        if random.random() < probLike:
                            self.network.addEdge(reader, article)
        if self.shouldHelp0DegreeUsers:
            self.help0DegreeUsers(iterations, article)
        if self.shouldHelp0DegreeArticles:
            self.help0DegreeArticles(iterations, self.network.users.values())
        # if iterations % 5 == 0:
        #     users = self.network.getUsersWithDegree0()
        #     for u in users:
        #         probLike = self.PLike(u, article)
        #         if random.random() < probLike:
        #             self.network.addEdge(u, article)
        self.runAnalysis(iterations)

