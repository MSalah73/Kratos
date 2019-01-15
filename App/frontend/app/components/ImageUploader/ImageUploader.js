import React from 'react';
import { Image, View, Text } from 'react-native';
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
    this.uploadImage(result.uri);
   }
 };

 uploadImage = async (uri) => {
   // In order for this to work, you will need connect your phone via usb and in the cmd write ipconfig .
   // This will list of ip. Look for the one thats named Ethernet adapter Eathernet # and copy it here.
   const apiUrl = 'http://IP address here:5000/predict';
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
     headers: {
       Accept: 'application/json',
       'Content-Type': 'multipart/form-data',
     },
   };

   return await fetch(apiUrl, options).
   then(response => response.json()).
   then(responseJson => this.setState({ text: "Prediction: "+responseJson.prediction }));
 };
}

export default ImageUploader;