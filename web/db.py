import boto3

class DB:
    def __init__(self):
        self.__dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2', aws_access_key_id='AKIARLSH5RYB4WKEG3UM', aws_secret_access_key= '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')

        self.__Cloth = self.__dynamodb.Table('Cloth')
        self.__User = self.__dynamodb.Table('User')
        self.__Cart = self.__dynamodb.Table('Cart')

    def GetAllCLothes(self):
        try:
            allData = self.__Cloth.scan()
            return allData['Items']
        except:
            print('error')
            return None

    def GetCloth(self, ID):
        try:
            item = self.__Cloth.get_item(Key = { 'ID': ID })
            return item['Item']
        except:
            print('error')
            return None

    def InsertCloth(self, ID, description, img_path):
        try:
            with self.__Cloth.batch_writer() as batch:
                batch.put_item(Item = { 'ID': ID, 'Description': description, 'ImgPath': img_path })
            return True
        except:
            print('error')
            return False

    def UpdateCloth(self, ID, new_description, new_img_path):
        try:
            self.__Cloth.update_item(
                Key = {
                    'ID': ID
                },
                UpdateExpression = 'SET Description = :new_description, ImgPath = :new_img_path',
                ExpressionAttributeValues = {
                    ':new_description': new_description,
                    ':new_img_path': new_img_path
                }
            )
            return True
        except:
            print('error')
            return False

    def DeleteCloth(self, ID):
        try:
            self.__Cloth.delete_item(
                Key = {
                    'ID': ID
                }
            )
            return True
        except:
            return False

    def GetAllUsers(self):
        try:
            allData = self.__User.scan()
            return allData['Items']
        except:
            print('error')
            return None

    def GetUser(self, ID):
        try:
            item = self.__User.get_item(Key = { 'ID': ID })
            return item['Item']
        except:
            print('error')
            return None

    def InsertUser(self, ID, pw, img_path):
        try:
            with self.__User.batch_writer() as batch:
                batch.put_item(Item = { 'ID': ID, 'password': pw, 'ImgPath': img_path })
            with self.__Cart.batch_writer() as batch:
                batch.put_item(Item = { 'UserID': ID, 'SelectedClothes': '' })
            return True
        except:
            print('error')
            return False

    def UpdateUser(self, ID, new_pw):
        try:
            self.__User.update_item(
                Key = {
                    'ID': ID
                },
                UpdateExpression = 'SET password = :new_pw',
                ExpressionAttributeValues = {
                    ':new_pw': new_pw,
                }
            )
            return True
        except:
            print('error')
            return False

    def DeleteUser(self, ID):
        try:
            if self.GetUser(ID) is None:
                return False

            self.__User.delete_item(
                Key = {
                    'ID': ID
                }
            )
            self.__Cart.delete_item(
                Key = {
                    'UserID': ID
                }
            )
            return True
        except:
            return False

    # return a list of items
    def GetUserCart(self, UserID):
        try:
            items = (self.__Cart.get_item(Key = { 'UserID': UserID }))['Item']['SelectedClothes']
            item_list = items.split()
            return item_list
        except:
            return None

    def InsertUserCart(self, UserID, ClothID):
        try:
            if self.GetUser(UserID) is None:
                return False
            if self.GetCloth(ClothID) is None:
                return False
            
            items = (self.__Cart.get_item(Key = { 'UserID': UserID }))['Item']['SelectedClothes']
            if ClothID in items:
                return False
            items += ' ' + ClothID

            self.__Cart.update_item(
                Key = {
                    'UserID': UserID
                },
                UpdateExpression = 'SET SelectedClothes = :new_SelectedClothes',
                ExpressionAttributeValues = {
                    ':new_SelectedClothes': items,
                }
            )
            return True
        except:
            return False

    def DeleteUserCart(self, UserID, ClothID):
        try:
            if self.GetUser(UserID) is None:
                return False
            if self.GetCloth(ClothID) is None:
                return False

            items = (self.__Cart.get_item(Key = { 'UserID': UserID }))['Item']['SelectedClothes']
            if ClothID in items:
                items = items.replace(' ' + ClothID, '')

                self.__Cart.update_item(
                    Key = {
                        'UserID': UserID
                    },
                    UpdateExpression = 'SET SelectedClothes = :new_SelectedClothes',
                    ExpressionAttributeValues = {
                        ':new_SelectedClothes': items,
                    }
                )
                return True
            else:
                return False
        except:
            return False

