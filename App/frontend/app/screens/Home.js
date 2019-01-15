import React from 'react';
import { View, StatusBar } from 'react-native';
import {Container} from '../components/Container'
import {Logo} from '../components/Logo';
import {ImageUploader} from '../components/ImageUploader';


export default () => (
	<Container>
		<StatusBar translucent={false} barStyle="light-content" />
		<Logo />
		<View />
		<ImageUploader />
	</Container>
);