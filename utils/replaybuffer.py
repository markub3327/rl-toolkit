import numpy as np
import pymongo

class ReplayBuffer:
    """
    An experiences replay buffer
    ----------------------

    Use database to store large amount of interactions from many games.
    Every game environment has own collection of documents stored in database.

    In-memory buffer is synchronized with database after every episode.

      *  MongoDB
      *  In-memory buffer (<= 1000)
    """

    def __init__(
        self, 
        env_name: str,
        server_name: str = "localhost",
        server_port: int = 27017,
        db_name: str = "rl-baselines"
    ):

        # connect to server
        self._client = pymongo.MongoClient(f"mongodb://{server_name}:{server_port}/")
        print(self._client.server_info())

        # select db
        if db_name in self._client.list_database_names():
            self._db = self._client[db_name]
        else:
            raise NameError(f"Database {db_name} was not found!")
        
        # select collection
        if env_name in self._db.list_collection_names():
            self._collection = self._db[env_name]
        else:
            raise NameError(f"Collection {env_name} was not found!")
        
        # drop all old data
        self._collection.remove({ })

        # in-memory experiences buffer
        self._buffer = []

    def store(self, state, action, reward, next_state, done, timestep):
        # transition (ID-State-Action-Reward-State-Done)
        self._buffer.append(dict(
            _id=timestep,        # the unique ID of document in collection <==> timestep in game !!!
            state=state.tobytes(),
            action=action.tobytes(),
            reward=reward,
            next_state=next_state.tobytes(),
            done=done
        ))

    def __len__(self):
        return len(self._buffer)

    def sync(self):
        # store transitions into db
        self._collection.insert_many(self._buffer)

        # clear buffer
        self._buffer.clear()