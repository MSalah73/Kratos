import React from 'react';
import { Image, View, Text, Header } from 'react-native';
import { Entypo } from '@expo/vector-icons';
import { ImagePicker, Permissions, FileSystem } from 'expo';
import styles from './styles';

class ImageUploader extends React.Component {
  state = {
    image: null,
    text: null,
  };


  render() {
    const { image, text } = this.state;

    return (
      <View style={styles.container}>
        <View style={styles.containerIcons}>
          <Entypo
            name="camera"
            size={28}
            onPress={this.cameraImageFetch}
          />
          <Entypo
            name="image-inverted"
            size={32}
            style={styles.containerIcon}
            onPress={this.libraryImageFetch}
          />
        </View>
        {image
	    	&& <Image source={{ uri: image }} style={styles.containerImage} />}
        <Text style={styles.containerText}>{text}</Text>
      </View>
    );
  }

 cameraImageFetch = async () => {
   const cameraPermission = await Permissions.askAsync(Permissions.CAMERA);
   const cameraRollPermission = await Permissions.askAsync(Permissions.CAMERA_ROLL);
   if (cameraPermission.status === 'granted' && cameraRollPermission.status === 'granted') {
     const result = await ImagePicker.launchCameraAsync({
       base64: true,
       aspect: [4, 3],
     });

     if (!result.cancelled) {
       this.setState({ image: result.uri });
       this.uploadImage(result.uri);
     }
   }
 };

 libraryImageFetch = async () => {
   const result = await ImagePicker.launchImageLibraryAsync({
     base64: true,
     aspect: [4, 3],
   });

   if (!result.cancelled) {
    this.setState({ image: result.uri });
    this.setState({text: null})
    this.uploadImage(result.uri);
   }
 };

 uploadImage = async (uri) => {
   // In order for this to work, you will need connect your phone via usb and in the cmd write ipconfig .
   // This will list of ip. Look for the one thats named Ethernet adapter Eathernet # and copy it here.
   const apiUrl = 'http://x.x.x.x:5000/predict';
   const uriParts = uri.split('.');
   const fileType = uriParts[uriParts.length - 1];

   const formData = new FormData();
   formData.append('photo', {
     uri,
     name: `photo.${fileType}`,
     type: `image/${fileType}`,
   });

   const options = {
     method: 'POST',
     body: formData,
     headendrs: {
       Accept: 'application/json',
       'Content-Type': 'multipart/form-data',
     },
   };
   return await fetch(apiUrl, options).
   then(response => response.json()).
   then(responseJson => {
     console.log(responseJson);
     this.setState({text: 'Predictions\n\n    Color:'+(responseJson.prediction[0].type === 'color'? responseJson.prediction[0].prediction.replace(/\W/g, ''): responseJson.prediction.message)+'\
     \n\nTop Five Categories:\n1st Model: '+(responseJson.prediction[1].type === 'category'? responseJson.prediction[1].prediction.join(): responseJson.prediction.message)+'\
     \n2nd Model: '+(responseJson.prediction[2].type === 'category'? responseJson.prediction[2].prediction.join(): responseJson.prediction.message)+'\
     \n3rd Model: '+(responseJson.prediction[3].type === 'category'? responseJson.prediction[3].prediction.join(): responseJson.prediction.message)+'\
     \n\nAttributes:\n'+(responseJson.prediction[4].type === 'Attributes'?responseJson.prediction[4].prediction[0].type: responseJson.prediction.message)
     +': '+ responseJson.prediction[4].prediction[0].prediction.join()+'\n'
     +responseJson.prediction[4].prediction[1].type+': '+responseJson.prediction[4].prediction[1].prediction.join()+'\n'
     +responseJson.prediction[4].prediction[2].type+': '+responseJson.prediction[4].prediction[2].prediction.join()+'\n'
     +responseJson.prediction[4].prediction[3].type+': '+responseJson.prediction[4].prediction[3].prediction.join()+'\n'
     +responseJson.prediction[4].prediction[4].type+': '+responseJson.prediction[4].prediction[4].prediction.join()})
     console.log(this.state.text);
   }).catch((error) => {console.error(error);});



 };
}

export default ImageUploader;
