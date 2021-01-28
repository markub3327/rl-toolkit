import numpy as np
import pymongo

from abc import ABC, abstractmethod


class ReplayBuffer(ABC):
    """
    Base class that represent an experiences replay buffer
    ----------------------

    Use database to store large amount of interactions from many games.
    Every game environment has own collection of documents stored in database.

    In-memory buffer is synchronized with database after every episode.

      *  MongoDB
      *  In-memory buffer (<= 1000)
    """

    def __init__(
        self,
        db_name: str,
        env_name: str,
        server_name: str,
        server_port: int = 27017,
    ):

        self.env_name = env_name
        self.db_name = db_name

        # connect to server
        self._client = pymongo.MongoClient(f"mongodb://{server_name}:{server_port}/")
        print(self._client.server_info())

        # select db
        if db_name in self._client.list_database_names():
            self._db = self._client[db_name]
        else:
            raise NameError(f"Database {db_name} was not found! 😕")

        # select collection
        if env_name in self._db.list_collection_names():
            self._collection = self._db[env_name]
        else:
            raise NameError(f"Collection {env_name} was not found! 😕")

        # drop all old data
        self._collection.remove({})

    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def sync(self):
        raise NotImplementedError()


class ReplayBufferWriter(ReplayBuffer):
    def __init__(
        self, db_name, env_name, size, obs_dim, act_dim, server_name, server_port=27017
    ):
        super().__init__(
            db_name, env_name, server_name=server_name, server_port=server_port
        )

        # in-memory experiences buffer
        self.obs_buf = np.zeros((size,) + obs_dim, dtype=np.float32)
        self.obs2_buf = np.zeros((size,) + obs_dim, dtype=np.float32)
        self.act_buf = np.zeros((size,) + act_dim, dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.size, self.max_size = 0, size

    def __len__(self):
        return self._collection.count()

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )

    def sync(self):
        db_count = self._collection.count()
        if db_count > self.max_size:
            query = self._collection.aggregate(
                [{"$sample": {"size": self.max_size}}]
            )  # num. of all samples
        else:
            query = self._collection.find()

        self.size = 0
        for x in query:
            self.obs_buf[self.size] = np.frombuffer(x["state"], dtype=np.float32)
            self.obs2_buf[self.size] = np.frombuffer(x["next_state"], dtype=np.float32)
            self.act_buf[self.size] = np.frombuffer(x["action"], dtype=np.float32)
            self.rew_buf[self.size] = x["reward"]
            self.done_buf[self.size] = x["done"]
            self.size += 1

        print(f"RPM max_size: {self.max_size}")
        print(f"RPM curr_size: {self.size}")


class ReplayBufferReader(ReplayBuffer):
    """
    An experiences replay buffer
    ----------------------
    Use database to store large amount of interactions from many games.
    Every game environment has own collection of documents stored in database.
    In-memory buffer is synchronized with database after every episode.
      *  MongoDB
      *  In-memory buffer (<= 1000)
    """

    def __init__(self, db_name, env_name, server_name, server_port=27017):
        super().__init__(
            db_name, env_name, server_name=server_name, server_port=server_port
        )

        # in-memory experiences buffer
        self._buffer = []

    def store(self, state, action, reward, next_state, done):
        # transition (ID-State-Action-Reward-State-Done)
        self._buffer.append(
            dict(
                # _id=timestep,  # the unique ID of document in collection <==> timestep in game !!!
                state=state.tobytes(),
                action=action.tobytes(),
                reward=reward,
                next_state=next_state.tobytes(),
                done=done,
            )
        )

    def __len__(self):
        return len(self._buffer)

    def sync(self):
        # store transitions into db
        self._collection.insert_many(self._buffer)

        # clear buffer
        self._buffer.clear()
