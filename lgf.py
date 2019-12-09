class NaiveLabelGeneratingFunction:
  def __init__(self):
    self.scorerWeights = {
      NaiveAdjectiveSentimentScorer(): 0.0,
      NaiveVerbSentimentScorer(): 0.0,
      IndicativeSentimentScorer(): 1.0
    }


  def label(self, testData):
    sentimentScore = 0.0
    for scorer in self.scorerWeights:
      sentimentScore += scorer.score(testData) * self.scorerWeights[scorer]
    if sentimentScore > 0:
      return 'pos'
    elif sentimentScore < 0:
      return 'neg'
    return None


class NaiveAdjectiveSentimentScorer:
  def __init__(self):
    self.goodWordScore = 1.0
    self.badWordScore = -3.0
    self.notGoodWordScore = -2.0 # correct for the score addition in goodWords
    self.goodWords = set([
      'good', 'best', 'great', 'awesome', 'perfect', 'clever', 'charming',
      'fascinating', 'pleasant', 'happy', 'hilarious', 'funny', 'wonderful', 'lovely'
    ])
    self.badWords = set([
      'bad', 'worst', 'horrible', 'terrible', 'stupid', 'boring', 'dreadful', 'disgust',
      'disturbing', 'problem', 'disaster', 'a waste', 'not a fan'
    ])
    self.notGoodWords = set()
    for goodWord in self.goodWords:
      self.notGoodWords.add('not ' + goodWord)
      self.notGoodWords.add('n\'t ' + goodWord)


  def score(self, s):
    s = s.lower()
    score = 0
    for goodWord in self.goodWords:
      if goodWord in s:
        score += self.goodWordScore
    for badWord in self.badWords:
      if badWord in s:
        score += self.badWordScore
    for notGoodWords in self.notGoodWords:
      if notGoodWords in s:
        score += self.notGoodWordScore
    return score


class NaiveVerbSentimentScorer:
  def __init__(self):
    self.goodVerbScore = 1.0
    self.badVerbScore = -3.0
    self.notGoodVerbScore = -2.0 # correct for the score addition in goodVerbs
    self.goodVerbs = set([
      'enjoy', 'like', 'love'
    ])
    self.badVerbs = set([
      'dislike', 'hate', 'waste', 'can\'t stand', 'can\'t tolerate'
    ])
    self.notGoodVerbs = set()
    for goodVerb in self.goodVerbs:
      self.notGoodVerbs.add('can\'t ' + goodVerb)
      self.notGoodVerbs.add('don\'t ' + goodVerb)


  def score(self, s):
    s = s.lower()
    score = 0
    for goodVerb in self.goodVerbs:
      if "i " + goodVerb in s:
        score += self.goodVerbScore
      if "we " + goodVerb in s:
        score += self.goodVerbScore
    for badVerb in self.badVerbs:
      if "i " + badVerb in s:
        score += self.badVerbScore
      if "we " + badVerb in s:
        score += self.badVerbScore
    for notGoodVerb in self.notGoodVerbs:
      if "i " + notGoodVerb in s:
        score += self.notGoodVerbScore
      if "we " + notGoodVerb in s:
        score += self.notGoodVerbScore
    return score


class IndicativeSentimentScorer:
  def __init__(self):
    self.goodWordScore = 1.0
    self.badWordScore = -1.0
    self.goodWords = set([
      "eddie", "stunning", "ship", "wonderful", "shakespeare", "henry", "hank", "finest", "professional", "watson", "fate", "con", "crowd", "germany", "mclaglen", "guilt", "crafted", "refreshing", "tremendous", "technology", "genuine", "groups", "jackie", "wonderfully", "sullivan", "favorites", "fay", "gothic", "hitman", "gorgeous", "captivating", "poignant", "segment", "teaches", "mafia", "kungfu", "stayed", "confronted", "perfection", "peace", "innocence", "immensely", "ethan", "expensive", "develops", "covered", "arrives", "gena", "superbly"
    ])
    self.badWords = set([
      "beaten", "pointless", "poorly", "laughable", "waste", "mediocre", "thugs", "310", "remotely", "amateurish", "cabin", "drags", "zombies", "worst", "bergman", "cardboard", "1972", "blatantly", "210", "accents", "garbage", "terrible", "awful", "wasting", "lowbudget", "horrible", "morality", "boll", "infected", "incomprehensible", "attack", "unwatchable", "painfully", "anne", "choosing", "flag", "horrendous", "forgettable", "jumps", "unfunny", "pack", "idiotic", "meaningless", "holiday", "zero", "bland", "moon", "crap", "zombie", "dire", "laid"
    ])


  def score(self, s):
    s = s.lower()
    score = 0
    for goodWord in self.goodWords:
      if goodWord in s:
        score += self.goodWordScore
    for badWord in self.badWords:
      if badWord in s:
        score += self.badWordScore
    return score